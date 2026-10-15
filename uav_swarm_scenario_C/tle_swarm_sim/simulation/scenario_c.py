"""Scenario C —— 物理自洽的固定监测区 + 过境预测性分布式孔径构型控制。

## 为什么必须重构（治本）

原仿真 `environment.build_satellite_task` 把真实星下点地面轨迹（550 km LEO，
480 s 内约 3100 km）线性缩放到 1 km 的局部参考域
（`track_limit = optimal_monitoring_radius_m(5000) * local_track_radius_fraction(0.2) = 1000 m`），
UAV 限速 20 m/s 才"追得上"。该缩放未在正文披露，导致"米级跟踪误差"不自洽。

## 但只去掉缩放会出现新的退化

550 km 处的卫星对一个半径 5 km 的地面区域：仰角变化 < 0.05 度、斜距变化 < 0.3 %；
卫星星下点距区域中心 300–1700 km，远超任何合理核半径。所以"靠近星下点"型目标
（原 `J_task`，核半径 2300 m）在真实尺度下对集群构型几乎不敏感——直接搬到真实
几何会让指标再次饱和（只是这次饱和在 0 而不是 1）。

## 本文件采用的物理目标：地面分布式孔径的波前对齐

地面 N 元阵列接收仰角 theta、方位 A 的卫星下行信号时，单元 i 相对参考点的相位差为

    phi_i = -(2 pi / lambda) * cos(theta) * (a_hat . p_i),   a_hat = 方位方向单位矢量

相干合成要求各单元同相 => 阵列必须**垂直于 a_hat**，即沿 u_hat = perp(a_hat) 方向
的投影方差要小。同时角分辨率要求基线展开，即沿 a_hat 方向的跨度要大。
故观测质量：

    G = ( sum_m w_m * exp( -Var_{i}(p_i . u_hat_m) / sigma_a^2 ) ) * min(1, span(v)/L_ref)

    u_hat_m  = 该星方位方向（星下点方向）的单位矢量
    v        = 阵列线方向
    w_m      ∝ clip(elev_m - elev_min, 0, 1)，即"优先观测谁"的权重

理想直线阵（沿 v 均匀展开 N 元、长 L_ref）沿 u_hat_m 的方差为
(L_ref^2/12) * (v . u_hat_m)^2，于是参考阵列方向取

    v*(t) = argmax_v  sum_m w_m * exp( -(L_ref^2/12) (v . u_hat_m)^2 / sigma_a^2 )

过境期间 u_hat_m 快速旋转，v*(t) 随之旋转——集群必须**预测性地**重定向这条孔径线。
这正是论文所称"预测势场梯度使集群提前机动"的物理版本，而且现在它是可验证的：
把 v*(t) 的实际旋转速率与各控制器的滞后对照即可。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tle_swarm_sim.sgp4_utils import ecef_to_geodetic, geodetic_to_ecef, satellite_ecef_at

from .environment import SatelliteTask

R_E = 6378137.0
POOL_FILENAME = "covisible_pool.json"
MIN_SUBPOINT_DIST_M = 300_000.0  # 排除天顶过境：天顶处方位角病态（keyhole）
MIN_VALID_FRACTION = 0.50
N_PHI = 181  # 阵列方向搜索栅格（半圆，1 度）


@dataclass(frozen=True, slots=True)
class SiteFrame:
    lat_deg: float
    lon_deg: float
    ecef_m: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray

    @classmethod
    def make(cls, lat_deg: float, lon_deg: float) -> "SiteFrame":
        la, lo = np.radians(lat_deg), np.radians(lon_deg)
        east = np.array([-np.sin(lo), np.cos(lo), 0.0])
        north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
        up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        return cls(
            lat_deg=float(lat_deg),
            lon_deg=float(lon_deg),
            ecef_m=geodetic_to_ecef(lat_deg, lon_deg, 0.0),
            east=east,
            north=north,
            up=up,
        )

    def enu(self, ecef_m: np.ndarray) -> tuple[float, float, float]:
        d = np.asarray(ecef_m, dtype=float) - self.ecef_m
        return float(d @ self.east), float(d @ self.north), float(d @ self.up)

    def subpoint_local_xy(self, record, ts: datetime) -> tuple[float, float, float]:
        """返回 (星下点东向 m, 北向 m, 仰角 deg)。

        星下点用 ENU 水平分量直接给出（等价于对星下点地表点做切平面投影，
        但少一次 ecef_to_geodetic 往返），仰角用 ENU 垂直/水平分量精确算。
        """
        sat_ecef = np.asarray(satellite_ecef_at(record, ts), dtype=float)
        e, n, u = self.enu(sat_ecef)
        elev = float(np.degrees(np.arctan2(u, np.hypot(e, n))))
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        gx, gy, _ = self.enu(geodetic_to_ecef(lat, lon, 0.0))
        return gx, gy, elev


_PHI = np.linspace(0.0, np.pi, N_PHI, endpoint=False)
_V_GRID = np.stack([np.cos(_PHI), np.sin(_PHI)], axis=1)  # (P,2) 候选阵列线方向


def pool_path(config) -> Path:
    raw = getattr(config, "aperture_pool_path", None)
    if raw:
        return Path(raw)
    return Path(config.repo_root) / "analysis" / POOL_FILENAME


def load_pool(config) -> dict:
    path = pool_path(config)
    if not path.exists():
        raise FileNotFoundError(f"缺少共视窗口池 {path}；请先运行 analysis/build_pool_sites.py")
    return json.loads(path.read_text(encoding="utf-8"))


def best_line_direction(priority: np.ndarray, u_hat: np.ndarray, config) -> tuple[np.ndarray, np.ndarray, float]:
    """给定优先级权重与各星方位方向，求参考阵列线方向。

    priority: (M,)   u_hat: (M,2)  ->  (v_star, 各候选方向质量 q(P,), q_star)
    """
    l_ref = float(config.aperture_baseline_m)
    sigma = float(config.aperture_sigma_m)
    # (L^2/12)*(v . u)^2 / sigma^2
    cosang = _V_GRID @ u_hat.T                       # (P, M)
    exponent = (l_ref**2 / 12.0) * cosang**2 / max(sigma**2, 1e-9)
    quality = (np.exp(-exponent) * priority[None, :]).sum(axis=1)   # (P,)
    best = int(np.argmax(quality))
    return _V_GRID[best], quality, float(quality[best])


def _rotate(vec: np.ndarray, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def rate_limited_directions(priority: np.ndarray, u_hat: np.ndarray, valid: np.ndarray,
                            config, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """瞬时最优方向 v_demand(t) 与受 UAV 限速约束的参考方向 v_ref(t)。

    参考直线阵长 L_ref，外缘单元到中心 L_ref/2；阵列转过角速率 omega 时外缘单元
    需要切向速度 (L_ref/2)*omega，不得超过 max_velocity_mps。故

        omega_max = max_velocity_mps / (L_ref/2)

    对 L_ref=3000 m、20 m/s 得 omega_max = 0.0133 rad/s = 0.763 deg/s。
    以该速率限幅跟随 v_demand，得到**动态可行**的参考构型；v_demand 与 v_ref 之间
    的差即"几何需求超出集群机动能力"的部分，单独作为指标报告。

    返回 (v_demand, v_ref, feasible_oracle, omega_max_rad_s)。
    """
    steps = priority.shape[0]
    l_ref = float(config.aperture_baseline_m)
    sigma = float(config.aperture_sigma_m)
    omega_max = float(config.max_velocity_mps) / max(0.5 * l_ref, 1e-9)

    v_demand = np.zeros((steps, 2), dtype=float)
    v_ref = np.zeros((steps, 2), dtype=float)
    oracle = np.zeros(steps, dtype=float)
    prev = np.array([0.0, 1.0])
    step_max = omega_max * dt
    for i in range(steps):
        if not bool(valid[i]):
            v_demand[i] = prev
            v_ref[i] = prev
            oracle[i] = np.nan
            continue
        v_t, _, _ = best_line_direction(priority[i], u_hat[i], config)
        # 线方向无向：取与上一参考同侧的分支
        if float(v_t @ prev) < 0.0:
            v_t = -v_t
        v_demand[i] = v_t
        ang = float(np.arccos(np.clip(float(v_t @ prev), -1.0, 1.0)))
        if ang <= step_max:
            prev = v_t
        else:
            cross = float(prev[0] * v_t[1] - prev[1] * v_t[0])
            prev = _rotate(prev, np.sign(cross) * step_max if cross != 0.0 else step_max)
        v_ref[i] = prev
        cosang = v_ref[i] @ u_hat[i].T
        oracle[i] = float((np.exp(-(l_ref**2 / 12.0) * cosang**2 / max(sigma**2, 1e-9)) * priority[i]).sum())
    return v_demand, v_ref, oracle, omega_max


def aperture_geometry(records, start_time: datetime, config) -> dict:
    steps = int(config.simulation_duration_s / config.control_step_s) + 1
    times = [start_time + timedelta(seconds=i * config.control_step_s) for i in range(steps)]
    frame = SiteFrame.make(config.site_lat_deg, config.site_lon_deg)
    m_count = len(records)

    sub_xy = np.zeros((steps, m_count, 2), dtype=float)
    elev = np.zeros((steps, m_count), dtype=float)
    for m, rec in enumerate(records):
        for i, ts in enumerate(times):
            gx, gy, el = frame.subpoint_local_xy(rec, ts)
            sub_xy[i, m] = (gx, gy)
            elev[i, m] = el
    dist = np.linalg.norm(sub_xy, axis=2)
    u_hat = sub_xy / np.maximum(dist, 1e-6)[..., None]

    min_elev = float(config.aperture_min_elevation_deg)
    w_raw = np.clip(elev - min_elev, 0.0, None)
    wsum = w_raw.sum(axis=1, keepdims=True)
    valid = wsum[:, 0] > 1e-9
    priority = np.where(wsum > 1e-9, w_raw / np.maximum(wsum, 1e-9), 0.0)

    v_demand, v_ref, oracle, omega_max = rate_limited_directions(
        priority, u_hat, valid, config, config.control_step_s)

    return {
        "steps": steps,
        "times": times,
        "frame": frame,
        "sub_xy": sub_xy,
        "elev_deg": elev,
        "sub_dist_m": dist,
        "u_hat": u_hat,
        "priority_w": priority,
        "v_demand": v_demand,
        "v_star": v_ref,
        "valid": valid,
        "valid_fraction": float(valid.mean()),
        "oracle_gain_step": oracle,
        "omega_max_rad_s": omega_max,
        "uav_speed_demand_limit_mps": float(0.5 * config.aperture_baseline_m) * omega_max,
        "site_lat_deg": config.site_lat_deg,
        "site_lon_deg": config.site_lon_deg,
    }


def reference_line(config) -> np.ndarray:
    """N 元理想直线阵的沿线坐标（沿 +x，中心在原点）。"""
    n = int(config.num_uavs)
    l_ref = float(config.aperture_baseline_m)
    if n == 1:
        return np.zeros(1, dtype=float)
    return (np.arange(n) - (n - 1) / 2.0) * (l_ref / (n - 1))


def build_aperture_task(records, start_time: datetime, config) -> tuple[SatelliteTask, dict]:
    geo = aperture_geometry(records, start_time, config)
    steps = geo["steps"]
    n = int(config.num_uavs)
    along = reference_line(config)

    positions = np.zeros((steps, n, 3), dtype=float)
    # 每个时刻沿 v_star(t) 摆放理想直线阵
    positions[:, :, 0] = along[None, :] * geo["v_star"][:, 0:1]
    positions[:, :, 1] = along[None, :] * geo["v_star"][:, 1:2]
    positions[:, :, 2] = config.uav_altitude_m
    velocities = np.gradient(positions, config.control_step_s, axis=0)

    orbital = np.zeros((n, 3), dtype=float)
    orbital[:, 0] = float(getattr(config, "orbital_height_m", 550000.0))
    orbital[:, 1] = float(getattr(config, "aperture_inclination_deg", 53.2))
    orbital[:, 2] = 0.0

    task = SatelliteTask(
        positions_m=positions,
        velocities_mps=velocities,
        weights=np.ones(n, dtype=float) / n,
        orbital_features=orbital,
    )
    return task, geo


# --------------------------------------------------------------------------
# 观测质量指标（与参考阵列解耦：只看实际构型的波前对齐与跨度）
# --------------------------------------------------------------------------
def aperture_quality(positions_xy: np.ndarray, geo: dict, step: int, config) -> dict:
    w = geo["priority_w"][step]
    u_hat = geo["u_hat"][step]          # (M,2)
    v = geo["v_star"][step]
    sigma = float(config.aperture_sigma_m)
    l_ref = float(config.aperture_baseline_m)

    s = positions_xy @ u_hat.T           # (N, M) 各单元沿各星方位方向的投影
    var_m = s.var(axis=0)                # (M,)
    align_m = np.exp(-var_m / max(sigma**2, 1e-9))
    alignment = float(np.sum(w * align_m))

    r = positions_xy @ v
    span = float(r.max() - r.min()) if r.size else 0.0
    coverage = float(min(1.0, span / max(l_ref, 1e-9)))

    centered = positions_xy - positions_xy.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(positions_xy.shape[0], 1)
    try:
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
    except np.linalg.LinAlgError:
        major = np.array([1.0, 0.0])
    cosang = abs(float(major @ v)) / max(float(np.linalg.norm(major)), 1e-9)
    misalign = float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
    if misalign > 90.0:
        misalign = 180.0 - misalign

    return {
        "aperture_gain": alignment * coverage,
        "aperture_alignment": alignment,
        "aperture_coverage": coverage,
        "aperture_wavefront_rms_m": float(np.sqrt(np.sum(w * var_m))),
        "aperture_span_m": span,
        "aperture_misalignment_deg": misalign,
    }


def _undirected_angles(v: np.ndarray) -> np.ndarray:
    if v.shape[0] < 2:
        return np.zeros(0, dtype=float)
    d = np.arccos(np.clip(np.sum(v[1:] * v[:-1], axis=1), -1.0, 1.0))
    return np.where(d > np.pi / 2, np.pi - d, d)


def aperture_summary(rows: list[dict], geo: dict, config) -> dict:
    valid = np.asarray(geo["valid"], dtype=bool)
    sel = [row for row, ok in zip(rows, valid) if ok] or list(rows)
    v_ref = geo["v_star"][valid] if bool(valid.any()) else geo["v_star"]
    v_dem = geo["v_demand"][valid] if bool(valid.any()) else geo["v_demand"]

    def _m(key: str) -> float:
        vals = [row[key] for row in sel if np.isfinite(row[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _tail(key: str, frac: float = 0.25) -> float:
        n = len(sel)
        tail = sel[int(n * (1.0 - frac)):] if n else []
        vals = [row[key] for row in tail if np.isfinite(row[key])]
        return float(np.mean(vals)) if vals else float("nan")

    d_ref = _undirected_angles(v_ref)
    d_dem = _undirected_angles(v_dem)
    oracle = geo["oracle_gain_step"][valid]
    oracle_mean = float(np.nanmean(oracle)) if oracle.size else float("nan")
    gain = _m("aperture_gain")
    dt = float(config.control_step_s)
    omega_max = float(geo.get("omega_max_rad_s", 0.0))
    return {
        "aperture_gain": gain,
        "aperture_gain_tail": _tail("aperture_gain"),
        "aperture_gain_min": float(np.min([r["aperture_gain"] for r in sel])) if sel else float("nan"),
        "aperture_alignment": _m("aperture_alignment"),
        "aperture_coverage": _m("aperture_coverage"),
        "aperture_wavefront_rms_m": _m("aperture_wavefront_rms_m"),
        "aperture_misalignment_deg": _m("aperture_misalignment_deg"),
        "aperture_misalignment_tail_deg": _tail("aperture_misalignment_deg"),
        "aperture_span_m": _m("aperture_span_m"),
        "aperture_gain_oracle": oracle_mean,
        "aperture_efficiency": float(gain / oracle_mean) if np.isfinite(oracle_mean) and oracle_mean > 0 else float("nan"),
        # 几何需求（不受限）与可行参考（限速后）
        "look_rate_demand_max_deg_s": float(np.degrees(d_dem.max()) / dt) if d_dem.size else 0.0,
        "look_rate_demand_mean_deg_s": float(np.degrees(np.mean(d_dem)) / dt) if d_dem.size else 0.0,
        "look_turn_demand_total_deg": float(np.degrees(np.sum(d_dem))) if d_dem.size else 0.0,
        "look_rate_ref_max_deg_s": float(np.degrees(d_ref.max()) / dt) if d_ref.size else 0.0,
        "look_rate_ref_mean_deg_s": float(np.degrees(np.mean(d_ref)) / dt) if d_ref.size else 0.0,
        "look_turn_ref_total_deg": float(np.degrees(np.sum(d_ref))) if d_ref.size else 0.0,
        "look_rate_limit_deg_s": float(np.degrees(omega_max)),
        "uav_tangential_speed_limit_mps": float(0.5 * config.aperture_baseline_m) * omega_max,
        "task_valid_fraction": float(valid.mean()),
        "subpoint_dist_min_km": float(geo["sub_dist_m"].min() / 1e3),
        "subpoint_dist_mean_km": float(geo["sub_dist_m"].mean() / 1e3),
        "elev_mean_deg": float(geo["elev_deg"][valid].mean()) if bool(valid.any()) else float("nan"),
    }


# --------------------------------------------------------------------------
# 窗口体检 / 选择
# --------------------------------------------------------------------------
_RANK_CACHE: dict[str, list[dict]] = {}


def rank_windows(records, config, sample_step_s: float = 10.0) -> list[dict]:
    key = f"{pool_path(config)}::{config.aperture_baseline_m}::{config.aperture_sigma_m}::{config.site_lon_deg}"
    if key in _RANK_CACHE:
        return _RANK_CACHE[key]

    pool = load_pool(config)
    frame = SiteFrame.make(pool["site_lat_deg"], pool["site_lon_deg"])
    by_cat = {str(r.catalog_number): r for r in records}
    window_s = float(getattr(config, "simulation_duration_s", 600.0))
    min_elev = float(getattr(config, "aperture_min_elevation_deg", 10.0))
    samples = int(window_s / sample_step_s) + 1

    ranked: list[dict] = []
    for item in pool["windows"]:
        cats = [c for c in item["catalogs"] if c in by_cat]
        if len(cats) < 2:
            continue
        start = datetime.fromisoformat(item["start_utc"])
        sub = np.zeros((len(cats), samples, 2), dtype=float)
        elev = np.zeros((len(cats), samples), dtype=float)
        for m, cat in enumerate(cats):
            for i in range(samples):
                gx, gy, el = frame.subpoint_local_xy(by_cat[cat], start + timedelta(seconds=i * sample_step_s))
                sub[m, i] = (gx, gy)
                elev[m, i] = el
        dist = np.linalg.norm(sub, axis=2)
        w_raw = np.clip(elev - min_elev, 0.0, None)
        wsum = w_raw.sum(axis=0, keepdims=True)
        valid = wsum[0] > 1e-9
        if int(valid.sum()) < 2 or float(valid.mean()) < MIN_VALID_FRACTION:
            continue
        priority = np.where(wsum > 1e-9, w_raw / np.maximum(wsum, 1e-9), 0.0)
        u_hat = sub / np.maximum(dist, 1e-6)[:, :, None]

        vs: list[np.ndarray | None] = []
        for i in range(samples):
            if not bool(valid[i]):
                vs.append(None)
                continue
            vs.append(best_line_direction(priority[:, i], u_hat[:, i], config)[0])
        v_idx = np.where(valid)[0]
        dphi = []
        for a, b in zip(v_idx[:-1], v_idx[1:]):
            if vs[a] is None or vs[b] is None:
                continue
            ang = np.arccos(np.clip(float(vs[a] @ vs[b]), -1.0, 1.0))
            if ang > np.pi / 2:
                ang = np.pi - ang
            dphi.append((ang, int(b - a)))
        if len(dphi) < 2:
            continue
        rate = float(max(a / g for a, g in dphi) / sample_step_s)
        ranked.append(
            {
                "start_utc": item["start_utc"],
                "catalogs": cats,
                "elev_start_deg": item["elev_start_deg"],
                "min_subpoint_km": float(dist.min() / 1e3),
                "max_look_rate_deg_s": float(np.degrees(rate)),
                "total_turn_deg": float(np.degrees(sum(a for a, _ in dphi))),
                "valid_fraction": float(valid.mean()),
            }
        )
    ranked.sort(key=lambda r: -r["max_look_rate_deg_s"])
    _RANK_CACHE[key] = ranked
    return ranked


def select_aperture_window(records, config, trial: int) -> dict:
    ranked = rank_windows(records, config)
    if not ranked:
        raise RuntimeError("共视窗口池里没有通过几何体检的窗口")
    return ranked[trial % len(ranked)]


def group_records_for_window(records, window: dict):
    by_cat = {str(r.catalog_number): r for r in records}
    return [by_cat[c] for c in window["catalogs"]]
