from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from tle_swarm_sim.sgp4_utils import ecef_to_geodetic, geodetic_to_ecef, satellite_ecef_at
from tle_swarm_sim.tle import TleRecord

from .config import SimulationConfig


@dataclass(frozen=True, slots=True)
class SatelliteTask:
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    weights: np.ndarray
    orbital_features: np.ndarray


def _surface_point_from_satellite(record: TleRecord, timestamp: datetime) -> np.ndarray:
    ecef = satellite_ecef_at(record, timestamp)
    lat_deg, lon_deg, _alt = ecef_to_geodetic(ecef)
    return geodetic_to_ecef(lat_deg, lon_deg, 0.0)


def _tangent_basis(site_lat_deg: float, site_lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.radians(site_lat_deg)
    lon = np.radians(site_lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=float)
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=float)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=float)
    return east, north, up


def _ecef_to_local_xy(ecef_m: np.ndarray, site_ecef_m: np.ndarray, site_lat_deg: float, site_lon_deg: float) -> np.ndarray:
    east, north, _up = _tangent_basis(site_lat_deg, site_lon_deg)
    delta = ecef_m - site_ecef_m
    return np.array([float(delta @ east), float(delta @ north)], dtype=float)


def build_satellite_task(
    records: list[TleRecord],
    start_time: datetime,
    site_lat_deg: float,
    site_lon_deg: float,
    config: SimulationConfig,
    weights: np.ndarray | None = None,
) -> SatelliteTask:
    steps = int(config.simulation_duration_s / config.control_step_s) + 1
    times = [start_time + timedelta(seconds=i * config.control_step_s) for i in range(steps)]
    site_ecef = geodetic_to_ecef(site_lat_deg, site_lon_deg, 0.0)
    track_limit = config.optimal_monitoring_radius_m * config.local_track_radius_fraction
    positions = np.zeros((steps, len(records), 3), dtype=float)
    orbital = np.zeros((len(records), 3), dtype=float)
    for s_idx, record in enumerate(records):
        from tle_swarm_sim.sgp4_utils import mean_orbit_altitude_m

        orbital[s_idx, 0] = mean_orbit_altitude_m(record)
        orbital[s_idx, 1] = record.inclination_deg
        orbital[s_idx, 2] = float(record.line2[17:25])
        raw_xy = np.zeros((steps, 2), dtype=float)
        for t_idx, ts in enumerate(times):
            surface = _surface_point_from_satellite(record, ts)
            raw_xy[t_idx] = _ecef_to_local_xy(surface, site_ecef, site_lat_deg, site_lon_deg)
        raw_xy -= raw_xy[0]
        max_norm = float(np.linalg.norm(raw_xy, axis=1).max())
        scale = min(1.0, track_limit / max(max_norm, 1.0))
        scaled_xy = raw_xy * scale
        positions[:, s_idx, :2] = scaled_xy
        for t_idx in range(steps):
            positions[t_idx, s_idx, 2] = config.uav_altitude_m
    velocities = np.gradient(positions, config.control_step_s, axis=0)
    if weights is None:
        weights = np.ones(len(records), dtype=float) / max(1, len(records))
    return SatelliteTask(positions_m=positions, velocities_mps=velocities, weights=weights, orbital_features=orbital)


def formation_offsets(num_uavs: int, num_satellites: int, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    assignment = np.arange(num_uavs) % max(1, num_satellites)
    offsets = np.zeros((num_uavs, 3), dtype=float)
    radius = getattr(config, "monitoring_slot_radius_m", config.formation_radius_m)
    for sat_idx in range(max(1, num_satellites)):
        members = np.where(assignment == sat_idx)[0]
        for rank, node_idx in enumerate(members):
            angle = 2.0 * np.pi * rank / max(1, len(members))
            offsets[node_idx, 0] = radius * np.cos(angle)
            offsets[node_idx, 1] = radius * np.sin(angle)
    return assignment, offsets


def monitoring_slots(task: SatelliteTask, step: int, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    n_ref = int(task.positions_m.shape[1])
    if n_ref == int(getattr(config, "num_uavs", -1)):
        # Scenario C：`positions_m` 本身就是逐机参考构型（理想直线阵），
        # 不存在"M 个 slot + 环形偏移"的结构，因此恒等指派、不做平滑
        # （参考构型已经是平滑的目标，再平滑会引入人为滞后）。
        return np.arange(n_ref), task.positions_m[step].copy()
    assignment, offsets = formation_offsets(config.num_uavs, n_ref, config)
    raw = task.positions_m[step, assignment] + offsets
    alpha = float(np.clip(getattr(config, "slot_smoothing_alpha", 1.0), 0.0, 1.0))
    if step <= 0 or alpha >= 1.0:
        return assignment, raw
    prev = task.positions_m[step - 1, assignment] + offsets
    smooth = alpha * raw + (1.0 - alpha) * prev
    smooth[:, 2] = config.uav_altitude_m
    return assignment, smooth


def initial_swarm_state(task: SatelliteTask, config: SimulationConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    assignment, offsets = formation_offsets(config.num_uavs, task.positions_m.shape[1], config)
    mode = getattr(config, "deployment_mode", "random_wide")
    if mode == "monitor_area_near":
        positions = np.zeros((config.num_uavs, 3), dtype=float)
        positions[:, :2] = rng.normal(0.0, config.monitor_area_initial_spread_m, size=(config.num_uavs, 2))
        positions[:, 2] = config.uav_altitude_m
    else:
        positions = task.positions_m[0, assignment] + offsets
        spread = config.warm_start_initial_spread_m if mode == "warm_start_near_popt" else config.initial_position_spread_m
        positions[:, :2] += rng.normal(0.0, spread, size=(config.num_uavs, 2))
    positions[:, 2] = config.uav_altitude_m
    velocities = rng.normal(0.0, 0.5, size=(config.num_uavs, 3))
    velocities[:, 2] = 0.0
    return positions, velocities
