from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tle_swarm_sim.simulation.config import SimulationConfig
from tle_swarm_sim.simulation.environment import SatelliteTask, formation_offsets
from tle_swarm_sim.simulation.network import adjacency_matrix


MAX_SATELLITES = 5
NODE_FEATURE_DIM = 71
PHYSICS_PRIOR_DIM = 16


@dataclass(frozen=True, slots=True)
class StgnnFeatureBatch:
    node_features: np.ndarray
    adjacency: np.ndarray
    physics_prior: np.ndarray
    assignment: np.ndarray
    feature_names: list[str]


def feature_names() -> list[str]:
    names = ["uav_px", "uav_py", "uav_pz", "uav_vx", "uav_vy", "uav_vz"]
    for sat_idx in range(MAX_SATELLITES):
        prefix = f"sat{sat_idx}"
        names.extend(
            [
                f"{prefix}_sx",
                f"{prefix}_sy",
                f"{prefix}_sz",
                f"{prefix}_svx",
                f"{prefix}_svy",
                f"{prefix}_svz",
                f"{prefix}_distance",
                f"{prefix}_task_weight",
                f"{prefix}_orbital_height",
                f"{prefix}_inclination",
                f"{prefix}_raan",
            ]
        )
    names.extend([f"assigned_sat_{i}" for i in range(MAX_SATELLITES)])
    names.extend([f"sat_mask_{i}" for i in range(MAX_SATELLITES)])
    return names


def _normalized_satellite_block(
    uav_pos: np.ndarray,
    task: SatelliteTask,
    step: int,
    sat_idx: int,
    config: SimulationConfig,
) -> list[float]:
    if sat_idx >= task.positions_m.shape[1]:
        return [0.0] * 11
    sat_pos = task.positions_m[step, sat_idx]
    sat_vel = task.velocities_mps[step, sat_idx]
    distance = float(np.linalg.norm(uav_pos - sat_pos))
    orbital = task.orbital_features[sat_idx] if task.orbital_features.size else np.zeros(3)
    return [
        sat_pos[0] / config.optimal_monitoring_radius_m,
        sat_pos[1] / config.optimal_monitoring_radius_m,
        sat_pos[2] / max(config.uav_altitude_m, 1.0),
        sat_vel[0] / max(config.max_velocity_mps, 1.0),
        sat_vel[1] / max(config.max_velocity_mps, 1.0),
        sat_vel[2] / max(config.max_velocity_mps, 1.0),
        distance / config.optimal_monitoring_radius_m,
        float(task.weights[sat_idx]),
        orbital[0] / 1_000_000.0,
        orbital[1] / 180.0,
        orbital[2] / 360.0,
    ]


def build_stgnn_features(
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    task: SatelliteTask,
    step: int,
    config: SimulationConfig,
    history_steps: int | None = None,
) -> StgnnFeatureBatch:
    history = history_steps or config.historical_observation_horizon
    steps = [max(0, step - (history - 1 - h)) for h in range(history)]
    assignment, _ = formation_offsets(config.num_uavs, task.positions_m.shape[1], config)
    names = feature_names()
    node_features = np.zeros((history, config.num_uavs, NODE_FEATURE_DIM), dtype=np.float32)
    for h_idx, h_step in enumerate(steps):
        for uav_id in range(config.num_uavs):
            feats: list[float] = [
                positions_m[uav_id, 0] / config.optimal_monitoring_radius_m,
                positions_m[uav_id, 1] / config.optimal_monitoring_radius_m,
                positions_m[uav_id, 2] / max(config.uav_altitude_m, 1.0),
                velocities_mps[uav_id, 0] / max(config.max_velocity_mps, 1.0),
                velocities_mps[uav_id, 1] / max(config.max_velocity_mps, 1.0),
                velocities_mps[uav_id, 2] / max(config.max_velocity_mps, 1.0),
            ]
            for sat_idx in range(MAX_SATELLITES):
                feats.extend(_normalized_satellite_block(positions_m[uav_id], task, h_step, sat_idx, config))
            feats.extend([1.0 if assignment[uav_id] == sat_idx else 0.0 for sat_idx in range(MAX_SATELLITES)])
            feats.extend([1.0 if sat_idx < task.positions_m.shape[1] else 0.0 for sat_idx in range(MAX_SATELLITES)])
            node_features[h_idx, uav_id, :] = np.asarray(feats, dtype=np.float32)
    adj = adjacency_matrix(positions_m, config).astype(np.float32)
    adjacency = np.repeat(adj[None, :, :], history, axis=0)
    prior = np.zeros(PHYSICS_PRIOR_DIM, dtype=np.float32)
    prior[0] = task.positions_m.shape[1] / MAX_SATELLITES
    prior[1 : 1 + MAX_SATELLITES] = [task.weights[i] if i < len(task.weights) else 0.0 for i in range(MAX_SATELLITES)]
    if task.orbital_features.size:
        prior[6] = float(np.mean(task.orbital_features[:, 0]) / 1_000_000.0)
        prior[7] = float(np.mean(task.orbital_features[:, 1]) / 180.0)
        prior[8] = float(np.mean(task.orbital_features[:, 2]) / 360.0)
    prior[9] = config.k_attr
    prior[10] = config.k_rep
    prior[11] = config.k_conn
    prior[12] = config.k_power
    prior[13] = config.rho
    prior[14] = config.max_acceleration_mps2
    prior[15] = config.max_velocity_mps
    return StgnnFeatureBatch(node_features=node_features, adjacency=adjacency, physics_prior=prior, assignment=assignment, feature_names=names)


def feature_sample_rows(batch: StgnnFeatureBatch, step: int, max_rows: int = 20) -> list[dict]:
    latest = batch.node_features[-1]
    rows = []
    for uav_id in range(min(max_rows, latest.shape[0])):
        row = {"step": step, "uav_id": uav_id, "assigned_satellite": int(batch.assignment[uav_id])}
        row.update({name: float(latest[uav_id, idx]) for idx, name in enumerate(batch.feature_names)})
        rows.append(row)
    return rows
