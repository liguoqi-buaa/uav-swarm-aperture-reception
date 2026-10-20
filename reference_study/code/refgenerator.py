"""Rate-constrained optimal reference generator.

Motivation
----------
In distributed-aperture reception the array must stay close to normal to the incoming
wave. The literature - including the earlier study behind this repository - constructs
the reference orientation as a *greedy follower*: at every step, turn toward the
instantaneous optimum ``v_demand`` at the maximum angular rate.

That construction has a structural defect, and it is the reason this module exists. Let

    omega_max = max_velocity_mps / (L_ref / 2)

be the angular rate available to the array (set by the tangential speed limit of its
outer elements). While saturated the follower satisfies

    theta_f(t) = theta_f(0) + omega_max * t

whereas the optimum advances as

    theta_star(t) = theta_star(0) + nu(t) * t .

Whenever ``|nu| > omega_max`` the lag

    theta_star - theta_f = (nu - omega_max) * t

grows **linearly and without bound**. Once the lag passes half a turn (pi) the reference
direction is arbitrary. Failing to keep up is therefore not a matter of degree: as soon
as the demand drifts faster than the limit, the reference detaches permanently.

Measured (equatorial site, 10 windows): the reference sits at the rate limit for 84% of
steps on average, in 3 of those windows for 100% of them, and those 3 windows reach a
reference quality of 0.20 against a reachable 0.82.

What this module does instead
-----------------------------
It treats the generation of the reference orientation as a rate-constrained optimal
control problem:

    max_{theta(0..T)}  sum_t  q(theta_t, t)
    s.t.               |d theta|_mod pi  <=  omega_max * dt

where ``q(theta, t) = sum_m w_m * exp(-(L^2/12) * cos^2(theta - phi_m) / sigma^2)`` is
the aperture quality of an ideal linear array at orientation ``theta``, defined exactly
as in the simulator's ``best_line_direction``.

The state is a single angle, a line direction is undirected (so the state space is a
circle of period pi), the cost is smooth and the constraint is a one-sided rate band.
With K angular grid points, T steps and at most +-D cells per step, backward dynamic
programming on the periodic circle gives the exact optimum at grid resolution - no
continuous optimiser involved.

Complexity O(T*K*D). At K=1800 (0.10 deg), T=601 and D~7 one window takes about 0.4 s
in plain NumPy, with no GPU.

Usage
-----
    from refgenerator import omega_max_for, optimal_reference

    omega = omega_max_for(max_velocity_mps=20.0, l_ref=3000.0)   # 1.333e-2 rad/s
    res = optimal_reference(phi, w, valid, omega, dt=1.0,
                            l_ref=3000.0, sigma=500.0, n_grid=1800)
    res.theta          # (T,) optimal orientation
    res.q              # (T,) quality at each step

    phi   : (T, M) satellite azimuths (rad), = atan2(u_hat[...,1], u_hat[...,0])
    w     : (T, M) priority weights (summing to 1, so q lies in [0, 1])
    valid : (T,)   whether a step carries a task; invalid steps score 0 but still obey
                   the rate constraint, so the reference stays continuous
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "omega_max_for",
    "quality_on_grid",
    "angle_of",
    "wrap_pi",
    "ReferenceResult",
    "optimal_reference",
    "reference_quality",
    "turn_stats",
]


# ------------------------------------------------------------------ 基本量

def omega_max_for(max_velocity_mps: float, l_ref: float) -> float:
    """外缘单元切向速度受限时的最大阵列角速率 (rad/s)。``omega_max = v_max / (L_ref/2)``。"""
    if l_ref <= 0.0:
        raise ValueError("l_ref 必须为正")
    return float(max_velocity_mps) / (0.5 * float(l_ref))


def quality_on_grid(theta: np.ndarray, phi: np.ndarray, w: np.ndarray,
                    l_ref: float, sigma: float) -> np.ndarray:
    """在朝向网格上求孔径质量。

        q(theta) = sum_m w_m * exp( -(L^2/12) * cos^2(theta - phi_m) / sigma^2 )

    与 ``scenario_c.best_line_direction`` 内的
        quality = (exp(-(l_ref**2/12) * cosang**2 / sigma**2) * priority).sum()
    完全一致（cosang = v·u_hat = cos(theta - phi)）。

    theta : (K,) 朝向网格；phi/w : (M,)；返回 (K,)
    """
    a = (float(l_ref) ** 2 / 12.0) / max(float(sigma) ** 2, 1e-9)
    d = np.asarray(theta, dtype=float)[:, None] - np.asarray(phi, dtype=float)[None, :]
    return np.exp(-a * np.cos(d) ** 2) @ np.asarray(w, dtype=float)


def angle_of(v: np.ndarray) -> np.ndarray:
    """方向向量 (...,2) -> 朝向角 mod pi（线方向无向，故取 pi 周期）。"""
    v = np.asarray(v, dtype=float)
    return np.mod(np.arctan2(v[..., 1], v[..., 0]), np.pi)


def wrap_pi(a: np.ndarray) -> np.ndarray:
    """把角差折到 (-pi/2, pi/2]，即无向线之间的最短夹角（绝对值）。"""
    a = np.abs(np.asarray(a, dtype=float))
    return np.minimum(a, np.pi - a)


# ------------------------------------------------------------------ 结果容器

@dataclass(slots=True)
class ReferenceResult:
    """最优参考的结果。"""
    theta: np.ndarray                     # (T,) 最优朝向 (rad, mod pi)
    q: np.ndarray                         # (T,) 每步质量
    q_total: float                        # sum_t q_t（DP 的目标值）
    n_grid: int
    resolution_deg: float
    reach_grid_steps: np.ndarray          # (T,) 每步可达网格步数 D_i
    max_turn_deg: float                   # 轨迹实际最大单步转角（自检用）
    diag: dict = field(default_factory=dict)


# ------------------------------------------------------------------ 核心 DP

def _dp_periodic(Q: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pi 周期圆环上的动态规划。

        V[i,k] = Q[i,k] + max_{|d| <= D[i]} V[i+1, (k+d) mod K]

    Q : (T, K)；D : (T,) 每步可达网格步数；返回 (最优索引轨迹 (T,), 每步质量 (T,))
    """
    T, K = Q.shape
    V = np.zeros(K, dtype=np.float64)                 # V[i+1]，末端初值 0
    nxt = np.zeros((T, K), dtype=np.int32)
    idx = np.arange(K, dtype=np.int32)
    for i in range(T - 1, -1, -1):
        best = V.copy()
        arg = idx.copy()
        for d in range(1, int(D[i]) + 1):
            for s in (+1, -1):
                sh = np.roll(V, -s * d)               # 周期边界由 roll 天然给出
                better = sh > best
                best = np.where(better, sh, best)
                arg = np.where(better, (idx + s * d) % K, arg)
        nxt[i] = arg
        V = Q[i] + best                               # 循环结束时 V 即 V[0]

    traj = np.zeros(T, dtype=np.int64)
    k = int(np.argmax(V))                             # 起点取总质量最大者
    for i in range(T):
        traj[i] = k
        k = int(nxt[i, k])
    return traj, Q[np.arange(T), traj]


def optimal_reference(phi: np.ndarray, w: np.ndarray, valid: np.ndarray,
                      omega_max: np.ndarray | float, dt: float,
                      l_ref: float, sigma: float,
                      n_grid: int = 1800) -> ReferenceResult:
    """求限速最优参考。

    phi : (T, M) 方位角；w : (T, M) 权重；valid : (T,)
    omega_max : 标量或 (T,) 每步速率上限 (rad/s)
    dt : 步长 (s)；l_ref / sigma : 孔径参数；n_grid : 角网格点数（[0, pi) 均分）
    """
    phi = np.asarray(phi, dtype=float)
    w = np.asarray(w, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    T = phi.shape[0]
    if phi.ndim != 2 or w.shape != phi.shape:
        raise ValueError("phi 与 w 必须是同形状的 (T, M)")
    if len(valid) != T:
        raise ValueError("valid 长度须等于 T")

    omega = np.broadcast_to(np.asarray(omega_max, dtype=float), (T,))
    delta = np.pi / n_grid
    theta = np.arange(n_grid) * delta

    # 质量表：无效步 q=0（不贡献目标，但速率约束仍施加，保证参考连续可跟踪）
    Q = np.zeros((T, n_grid), dtype=np.float64)
    for i in range(T):
        if bool(valid[i]):
            Q[i] = quality_on_grid(theta, phi[i], w[i], l_ref, sigma)

    # 逐步可达网格步数，取 floor 保证 D*delta <= omega*dt（严格可行，绝不越限）
    D = np.floor(omega * dt / delta + 1e-12).astype(np.int64)

    traj, q_traj = _dp_periodic(Q, D)
    theta_opt = traj * delta

    # 自检：实际最大单步转角必须落在速率上限内
    if T > 1:
        d = wrap_pi(np.diff(theta_opt))
        max_turn = float(np.degrees(d.max()))
    else:
        max_turn = 0.0
    limit = float(np.degrees(omega.max() * dt))

    return ReferenceResult(
        theta=theta_opt,
        q=q_traj,
        q_total=float(np.nansum(np.where(valid, q_traj, 0.0))),
        n_grid=n_grid,
        resolution_deg=float(np.degrees(delta)),
        reach_grid_steps=D,
        max_turn_deg=max_turn,
        diag={"rate_limit_deg": limit,
              "max_turn_within_limit": bool(max_turn <= limit + 1e-6),
              "n_invalid": int((~valid).sum())},
    )


# ------------------------------------------------------------------ 评估工具

def reference_quality(theta: np.ndarray, phi: np.ndarray, w: np.ndarray,
                      valid: np.ndarray, l_ref: float, sigma: float) -> np.ndarray:
    """给定任意参考朝向轨迹，逐点求质量（无效步返回 nan）。"""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    w = np.asarray(w, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    q = np.array([
        quality_on_grid(np.array([theta[i]]), phi[i], w[i], l_ref, sigma)[0]
        if bool(valid[i]) else np.nan
        for i in range(len(theta))
    ])
    return q


def turn_stats(theta: np.ndarray, valid: np.ndarray, step_max_rad: float) -> dict:
    """参考轨迹的转动统计：中位/P90 单步转角，以及速率饱和步占比。

    饱和步占比是识别"贪心跟随者"的关键诊断量：接近 1 说明该参考每一步都吃满速率、
    即始终处于追赶状态（很可能存在无界滞后）。
    """
    theta = np.asarray(theta, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if len(theta) < 2:
        return {"median_deg": 0.0, "p90_deg": 0.0, "sat_frac": 0.0}
    d = wrap_pi(np.diff(theta))
    m = valid[:-1] & valid[1:]
    d = d[m] if bool(m.any()) else d
    thr = 0.999 * float(step_max_rad)
    return {
        "median_deg": float(np.degrees(np.median(d))) if d.size else 0.0,
        "p90_deg": float(np.degrees(np.percentile(d, 90))) if d.size else 0.0,
        "sat_frac": float(np.mean(d > thr)) if d.size else 0.0,
    }


def lag_fraction(theta_ref: np.ndarray, theta_demand: np.ndarray,
                 valid: np.ndarray, threshold_deg: float = 30.0) -> float:
    """参考与瞬时最优夹角超过阈值的（有效步）占比。论文 §4.3 报的是 33%。"""
    theta_ref = np.asarray(theta_ref, dtype=float)
    theta_demand = np.asarray(theta_demand, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    gap = np.degrees(wrap_pi(theta_ref - theta_demand))
    return float(np.mean(gap[valid] > threshold_deg)) if bool(valid.any()) else float("nan")
