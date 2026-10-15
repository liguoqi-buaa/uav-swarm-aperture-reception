"""独立 baseline 控制器（与主控制律无关）。

三族前瞻方式（`kind`）
--------------------
- ``none``  : 使用 `step-1` 时刻的几何（与主方法同响应时延），**无前瞻**。
- ``shift`` : 目标直接取 `step-1+H` 时刻的几何（简化前瞻，非最优）。
- ``prev``  : **滚动时域预览控制**——在每个控制步求解一个含加速度/速度双约束的 QP，
  在 H 步时域内跟踪未来目标轨迹，只执行第一步。这是"把前瞻做对"的版本。

目标朝向模式（`mode`）
--------------------
- ``perp``   : 垂直于"当前加权卫星方向"的直线阵（朴素几何）。
- ``greedy`` : 瞬时 A 最优朝向 `v_demand`。

公平性设计
----------
1. 所有变体使用 `step-1` 作为"当前"信息截点（与主方法 `response_delay_steps=1` 一致），
   因此"有没有前瞻"是唯一的结构性差别。
2. 共享同一套物理：二阶动力学、速度/加速度上限、通信模型、孔径度量。
3. 名字后缀可指定增益（PD 版 `__kp<int>_kd<int>`）或预览权重档位（预浏览 `__w<int>`），
   便于做"对 baseline 有利"的调参扫描。

命名
----
    <BASE>
    <BASE>__kp<int>_kd<int>     # PD 版增益
    <BASE>__w<int>              # 预览版权重档位（见 PREVIEW_WEIGHTS）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .config import SimulationConfig

LOOKAHEAD_STEPS = 5        # 与主方法 prediction_horizon 一致
RESPONSE_DELAY_STEPS = 1   # 与主方法 response_delay_steps 一致
DEFAULT_KP = 1.0
DEFAULT_KD = 2.0

# (wp, wv, wa)：目标位置权重、目标速度权重、控制力度权重
PREVIEW_WEIGHTS: list[tuple[float, float, float]] = [
    (1.0, 1.0, 0.05),
    (1.0, 4.0, 0.02),
    (1.0, 9.0, 0.01),
]

# name -> (target mode, lookahead kind)
INDEPENDENT_BASELINES: dict[str, tuple[str, str]] = {
    "B1_GEOM_PD":        ("perp", "none"),
    "B2_GREEDY":         ("greedy", "none"),
    "B1_GEOM_PD_H5":     ("perp", "shift"),
    "B2_GREEDY_H5":      ("greedy", "shift"),
    "B1_GEOM_PD_PREV":   ("perp", "prev"),
    "B2_GREEDY_PREV":    ("greedy", "prev"),
}

_SUFFIX = re.compile(
    r"^(?P<base>[A-Z0-9_]+?)"
    r"(?:__kp(?P<kp>\d+)_kd(?P<kd>\d+)|__w(?P<w>\d+))?$"
)


@dataclass(frozen=True, slots=True)
class IndependentSpec:
    name: str
    mode: str
    kind: str
    kp: float
    kd: float
    weights: tuple[float, float, float]


def parse_independent(algorithm: str) -> IndependentSpec | None:
    """若 algorithm 属于独立 baseline，返回其规格；否则返回 None。"""
    m = _SUFFIX.match(algorithm)
    if not m:
        return None
    spec = INDEPENDENT_BASELINES.get(m.group("base"))
    if spec is None:
        return None
    mode, kind = spec
    kp = float(m.group("kp")) if m.group("kp") else DEFAULT_KP
    kd = float(m.group("kd")) if m.group("kd") else DEFAULT_KD
    w_idx = int(m.group("w")) if m.group("w") else 0
    w_idx = min(max(w_idx, 0), len(PREVIEW_WEIGHTS) - 1)
    return IndependentSpec(name=algorithm, mode=mode, kind=kind,
                           kp=kp, kd=kd, weights=PREVIEW_WEIGHTS[w_idx])


# ---------------------------------------------------------------------------
# 目标几何
# ---------------------------------------------------------------------------
def _target_direction(spec: IndependentSpec, geometry: dict, idx: int) -> np.ndarray:
    if spec.mode == "perp":
        w = np.asarray(geometry["priority_w"][idx], dtype=float)
        u = np.asarray(geometry["u_hat"][idx], dtype=float)
        ubar = (w[:, None] * u).sum(axis=0)
        nrm = float(np.linalg.norm(ubar))
        if nrm < 1e-9:
            ubar = u.mean(axis=0)
            nrm = max(float(np.linalg.norm(ubar)), 1e-9)
        ubar = ubar / nrm
        v = np.array([-ubar[1], ubar[0]], dtype=float)
    else:  # greedy：瞬时 A 最优朝向
        v = np.asarray(geometry["v_demand"][idx], dtype=float).copy()
    nrm = float(np.linalg.norm(v))
    return v / max(nrm, 1e-9)


def _slot_offsets(n_uav: int, l_ref: float) -> np.ndarray:
    return (np.arange(n_uav) - (n_uav - 1) / 2.0) * (l_ref / max(n_uav - 1, 1))


def _target_positions(spec, geometry, idx, n_uav, l_ref) -> np.ndarray:
    v = _target_direction(spec, geometry, idx)
    return _slot_offsets(n_uav, l_ref)[:, None] * v[None, :]


# ---------------------------------------------------------------------------
# PD（无前瞻 / 简化前瞻）
# ---------------------------------------------------------------------------
def pd_accel(spec, positions_m, velocities_mps, geometry, idx, config) -> np.ndarray:
    n_uav = positions_m.shape[0]
    l_ref = float(config.aperture_baseline_m)
    dt = float(config.control_step_s)
    tgt = _target_positions(spec, geometry, idx, n_uav, l_ref)
    tgt_prev = _target_positions(spec, geometry, max(0, idx - 1), n_uav, l_ref)
    v_tgt = (tgt - tgt_prev) / dt

    accel = np.zeros_like(positions_m)
    accel[:, :2] = spec.kp * (tgt - positions_m[:, :2]) + spec.kd * (v_tgt - velocities_mps[:, :2])
    return accel


# ---------------------------------------------------------------------------
# 滚动时域预览控制（QP）
# ---------------------------------------------------------------------------
def _preview_matrices(H: int, dt: float, wp: float, wv: float, wa: float):
    """返回 (C, D, P_blk, Dcon)。a 为 (H,) 决策变量。"""
    C = np.zeros((H, H))
    D = np.zeros((H, H))
    for k in range(1, H + 1):
        for j in range(0, k):
            C[k - 1, j] = (k - j - 0.5)
            D[k - 1, j] = 1.0
    P = 2.0 * (wp * dt ** 4 * (C.T @ C) + wv * dt ** 2 * (D.T @ D) + wa * np.eye(H))
    return C, D, P, D


@lru_cache(maxsize=8)
def _build_qp(n_uav: int, H: int, dt: float, weights: tuple, a_max: float, v_max: float):
    """构造块对角 QP 的常量部分（P、约束矩阵、约束上下界的常量部分）。"""
    wp, wv, wa = weights
    C, D, P_blk, _ = _preview_matrices(H, dt, wp, wv, wa)
    nblk = n_uav * 2                       # (uav, axis) 块
    A = sp.block_diag([sp.vstack([sp.identity(H), D]) for _ in range(nblk)], format="csc")
    # 上块：a 的箱式约束；下块：速度约束（依赖 v0，每步更新）
    return C, D, P_blk, A, nblk


_QP_CACHE: dict = {}


def _get_solver(n_uav, H, dt, weights, a_max, v_max):
    import osqp

    key = (n_uav, H, dt, tuple(weights), a_max, v_max)
    if key in _QP_CACHE:
        return _QP_CACHE[key]
    C, D, P_blk, A, nblk = _build_qp(n_uav, H, dt, tuple(weights), a_max, v_max)
    P = sp.block_diag([sp.csc_matrix(P_blk) for _ in range(nblk)], format="csc")
    q = np.zeros(nblk * H)
    nb = nblk * H
    l = np.full(nb, -a_max)
    u = np.full(nb, a_max)
    lv = np.full(nblk * H, -1e9)
    uv = np.full(nblk * H, 1e9)
    l2 = np.concatenate([l, lv])
    u2 = np.concatenate([u, uv])
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l2, u=u2, verbose=False, warm_start=True,
               eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polish=False)
    _QP_CACHE[key] = (prob, C, D, P_blk, nblk)
    return _QP_CACHE[key]


def preview_accel(spec, positions_m, velocities_mps, geometry, idx0, config) -> np.ndarray:
    """滚动时域预览：跟踪未来 H 步的目标轨迹，含加速度/速度双约束。"""
    n_uav = positions_m.shape[0]
    H = LOOKAHEAD_STEPS
    dt = float(config.control_step_s)
    l_ref = float(config.aperture_baseline_m)
    a_max = float(config.max_acceleration_mps2)
    v_max = float(config.max_velocity_mps)
    wp, wv, wa = spec.weights
    steps_total = np.asarray(geometry["u_hat"]).shape[0]

    prob, C, D, P_blk, nblk = _get_solver(n_uav, H, dt, spec.weights, a_max, v_max)

    # 目标轨迹 (N, H, 2) 与目标速度 (N, H, 2)
    offs = _slot_offsets(n_uav, l_ref)
    dirs = np.zeros((H + 1, 2))
    for h in range(H + 1):
        dirs[h] = _target_direction(spec, geometry, int(np.clip(idx0 + h, 0, steps_total - 1)))
    tgt = np.einsum("n,hk->nhk", offs, dirs[1:])        # k = 1..H
    tgt_prev = np.einsum("n,hk->nhk", offs, dirs[:-1])  # k = 0..H-1
    v_tgt = (tgt - tgt_prev) / dt

    # 组装 q：按 (uav, axis) 块顺序
    ks = np.arange(1, H + 1, dtype=float)[:, None]      # (H,1)
    q_blocks = []
    v0_blocks = []
    for ax in range(2):
        p0 = positions_m[:, ax][:, None]                 # (N,1)
        v0 = velocities_mps[:, ax][:, None]              # (N,1)
        pt = tgt[:, :, ax]                               # (N,H)
        vt = v_tgt[:, :, ax]                             # (N,H)
        r = (p0 + ks.T * v0 * dt) - pt                   # (N,H)
        s = v0 - vt                                      # (N,H)
        # 注意：需 (C^T r)_k = sum_m C[m,k] r[m]，即 r @ C（C 为下三角）
        q = 2.0 * (wp * dt ** 2 * (r @ C) + wv * dt * (s @ D))   # (N,H)
        q_blocks.append(q)
        v0_blocks.append(v0)
    # 交错顺序：块索引 = n*2 + ax；为与 block_diag 对齐，按 (n, ax) 排列
    q_all = np.zeros((n_uav, 2, H))
    v0_all = np.zeros((n_uav, 2))
    for ax in range(2):
        q_all[:, ax, :] = q_blocks[ax]
        v0_all[:, ax] = v0_blocks[ax][:, 0]
    q_vec = q_all.reshape(-1)

    # 速度约束上下界： -v_max <= v0 + dt*D a <= v_max
    lv = (-v_max - v0_all[:, :, None].repeat(H, axis=2)) / dt
    uv = (v_max - v0_all[:, :, None].repeat(H, axis=2)) / dt
    l_viol = np.full(nblk * H, -a_max)
    u_viol = np.full(nblk * H, a_max)
    l2 = np.concatenate([l_viol, lv.reshape(-1)])
    u2 = np.concatenate([u_viol, uv.reshape(-1)])

    prob.update(q=q_vec, l=l2, u=u2)
    res = prob.solve()
    a = res.x.reshape(n_uav, 2, H)

    accel = np.zeros_like(positions_m)
    accel[:, :2] = a[:, :, 0]
    return accel


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def _apply_limits(accel, positions_m, velocities_mps, config) -> np.ndarray:
    alt = float(getattr(config, "uav_altitude_m", 300.0))
    accel[:, 2] = -0.5 * (positions_m[:, 2] - alt) - 1.0 * velocities_mps[:, 2]
    a_max = float(config.max_acceleration_mps2)
    norms = np.linalg.norm(accel[:, :2], axis=1, keepdims=True)
    xy = np.where(norms > a_max, accel[:, :2] / np.maximum(norms, 1e-9) * a_max, accel[:, :2])
    accel[:, :2] = xy
    accel[:, 2] = np.clip(accel[:, 2], -0.5, 0.5)
    return accel


def independent_control(spec, positions_m, velocities_mps, step, config, geometry) -> np.ndarray:
    steps_total = np.asarray(geometry["u_hat"]).shape[0]
    if spec.kind == "none":
        idx = int(np.clip(step - RESPONSE_DELAY_STEPS, 0, steps_total - 1))
        accel = pd_accel(spec, positions_m, velocities_mps, geometry, idx, config)
    elif spec.kind == "shift":
        idx = int(np.clip(step - RESPONSE_DELAY_STEPS + LOOKAHEAD_STEPS, 0, steps_total - 1))
        accel = pd_accel(spec, positions_m, velocities_mps, geometry, idx, config)
    else:  # prev
        idx0 = int(np.clip(step - RESPONSE_DELAY_STEPS, 0, steps_total - 1))
        accel = preview_accel(spec, positions_m, velocities_mps, geometry, idx0, config)
    return _apply_limits(accel, positions_m, velocities_mps, config)
