from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .environment import SatelliteTask, formation_offsets, monitoring_slots
from .network import adjacency_matrix, algebraic_connectivity


@dataclass(frozen=True, slots=True)
class PotentialResult:
    total: float
    attraction: float
    repulsion: float
    connectivity: float
    power: float
    gradient: np.ndarray
    lambda2: float


def optimal_positions(task: SatelliteTask, step: int, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    sat_count = task.positions_m.shape[1]
    if sat_count == int(getattr(config, "num_uavs", -1)):
        # Scenario C：逐机参考构型，恒等指派（见 environment.monitoring_slots）
        assignment = np.arange(sat_count)
        opt = np.repeat(task.positions_m[step][:, None, :], sat_count, axis=1)
        return assignment, opt
    assignment, offsets = formation_offsets(config.num_uavs, sat_count, config)
    opt = np.zeros((config.num_uavs, sat_count, 3), dtype=float)
    for m in range(sat_count):
        opt[:, m, :] = task.positions_m[step, m] + offsets
    alpha = float(np.clip(getattr(config, "slot_smoothing_alpha", 1.0), 0.0, 1.0))
    if step > 0 and alpha < 1.0:
        prev = np.zeros_like(opt)
        for m in range(sat_count):
            prev[:, m, :] = task.positions_m[step - 1, m] + offsets
        opt = alpha * opt + (1.0 - alpha) * prev
        opt[:, :, 2] = config.uav_altitude_m
    return assignment, opt


def monitoring_performance(positions_m: np.ndarray, task: SatelliteTask, step: int, config: SimulationConfig) -> float:
    assignment, opt = optimal_positions(task, step, config)
    uav_xy = positions_m[:, :2]
    radius = config.monitoring_kernel_radius_m
    exponent = config.monitoring_kernel_exponent
    per_sat = np.zeros(task.positions_m.shape[1], dtype=float)
    for sat_idx in range(task.positions_m.shape[1]):
        members = np.where(assignment == sat_idx)[0]
        if members.size == 0:
            continue
        diff = uav_xy[members] - opt[members, sat_idx, :2]
        dist = np.linalg.norm(diff, axis=1)
        per_sat[sat_idx] = np.exp(-float(np.mean((dist / radius) ** exponent)))
    return float(task.weights @ per_sat)


def tracking_error(positions_m: np.ndarray, task: SatelliteTask, step: int, config: SimulationConfig) -> float:
    assignment, assigned_opt = monitoring_slots(task, step, config)
    return float(np.linalg.norm(positions_m - assigned_opt, axis=1).mean())


def composite_potential(
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    task: SatelliteTask,
    step: int,
    config: SimulationConfig,
    gains_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> PotentialResult:
    k_attr = config.k_attr * gains_scale[0]
    k_rep = config.k_rep * gains_scale[1]
    k_conn = config.k_conn * gains_scale[2]
    k_power = config.k_power * gains_scale[3]

    assignment, assigned_opt = monitoring_slots(task, step, config)
    assigned_weights = task.weights[assignment]
    diff = positions_m - assigned_opt
    attraction = 0.5 * k_attr * float(np.sum(assigned_weights[:, None] * diff * diff))
    grad = k_attr * assigned_weights[:, None] * diff

    repulsion = 0.0
    for i in range(config.num_uavs):
        for j in range(i + 1, config.num_uavs):
            vector = positions_m[i] - positions_m[j]
            dist = float(np.linalg.norm(vector))
            if 1e-6 < dist < config.safe_distance_m:
                margin = config.safe_distance_m - dist
                repulsion += 0.5 * k_rep * margin * margin
                pair_grad = -k_rep * margin * vector / dist
                grad[i] += pair_grad
                grad[j] -= pair_grad

    adj = adjacency_matrix(positions_m, config)
    lambda2 = algebraic_connectivity(adj)
    connectivity = 0.5 * k_conn * max(0.0, config.lambda_threshold - lambda2) ** 2
    if lambda2 < config.lambda_threshold:
        centroid = positions_m.mean(axis=0)
        grad += k_conn * (positions_m - centroid) / max(1.0, config.max_communication_range_m)

    power = 0.5 * k_power * float(np.sum(velocities_mps * velocities_mps))
    grad += k_power * velocities_mps
    grad[:, 2] += 0.5 * (positions_m[:, 2] - config.uav_altitude_m)

    total = attraction + repulsion + connectivity + power
    return PotentialResult(
        total=total,
        attraction=attraction,
        repulsion=repulsion,
        connectivity=connectivity,
        power=power,
        gradient=grad,
        lambda2=lambda2,
    )
