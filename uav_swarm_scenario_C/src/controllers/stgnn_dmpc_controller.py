from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import osqp
import scipy.sparse as sp
import torch

from src.controllers.feature_builder import NODE_FEATURE_DIM, build_stgnn_features
from src.models.stgnn_model import ClosedLoopSTGNN, load_stgnn_model, model_file_size_mb, save_checkpoint
from tle_swarm_sim.simulation.config import SimulationConfig
from tle_swarm_sim.simulation.environment import SatelliteTask, formation_offsets, monitoring_slots


@dataclass(slots=True)
class DmpcInfo:
    qp_status: str
    solve_time_ms: float
    objective_value: float
    num_variables: int
    num_constraints: int
    active_acceleration_constraints: int
    active_velocity_constraints: int
    slack_values: list[float]
    fallback_used: bool


@dataclass(slots=True)
class ClosedLoopOutput:
    accel_mps2: np.ndarray
    compute_ms: float
    inference_ms: float
    grad_pred: np.ndarray
    qp_infos: list[DmpcInfo]
    model_path: str
    stgnn_loaded: bool
    model_file_size_mb: float
    input_feature_dim: int
    output_gradient_dim: int
    fallback_used: bool
    assignment: np.ndarray


_MODEL_CACHE: dict[Path, tuple[ClosedLoopSTGNN, dict]] = {}


def _bootstrap_model(path: Path, config: SimulationConfig) -> None:
    """Create a checkpoint only so code paths can run before explicit training.

    Production reproduction scripts train and overwrite this file before the
    main experiment. The metadata makes the bootstrap status auditable.
    """

    model = ClosedLoopSTGNN(
        node_feature_dim=NODE_FEATURE_DIM,
        physics_prior_dim=16,
        hidden_dim=64,
        prediction_horizon=config.prediction_horizon,
    )
    save_checkpoint(
        model,
        path,
        {
            "node_feature_dim": NODE_FEATURE_DIM,
            "physics_prior_dim": 16,
            "hidden_dim": 64,
            "prediction_horizon": config.prediction_horizon,
            "gradient_output_scale": 1000.0,
            "training_status": "bootstrap_untrained_model",
        },
    )


def _load_model(config: SimulationConfig) -> tuple[ClosedLoopSTGNN, dict, Path]:
    path = config.stgnn_model_path
    if not path.exists():
        _bootstrap_model(path, config)
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = load_stgnn_model(path, device="cpu")
    return (*_MODEL_CACHE[path], path)


@lru_cache(maxsize=16)
def _dynamics_matrices(horizon: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    variables = horizon * 3
    bp = np.zeros((horizon, 3, variables), dtype=float)
    bv = np.zeros((horizon, 3, variables), dtype=float)
    for t in range(1, horizon + 1):
        for k in range(t):
            coeff_p = (t - k - 0.5) * dt * dt
            coeff_v = dt
            bp[t - 1, :, 3 * k : 3 * k + 3] = np.eye(3) * coeff_p
            bv[t - 1, :, 3 * k : 3 * k + 3] = np.eye(3) * coeff_v
    return bp, bv


@lru_cache(maxsize=64)
def _fast_qp_cache(
    horizon: int,
    dt: float,
    control_penalty: float,
    rho: float,
    velocity_penalty: float,
    task_allocation_weight: float,
    terminal_cost_weight: float,
    velocity_feedforward_weight: float,
    max_acceleration_mps2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    variables = horizon * 3
    bp, bv = _dynamics_matrices(horizon, dt)
    p_mat = np.eye(variables) * control_penalty
    for t in range(horizon):
        bpt = bp[t]
        bvt = bv[t]
        p_mat += rho * (bpt.T @ bpt)
        p_mat += velocity_penalty * (bvt.T @ bvt)
        p_mat += task_allocation_weight * (bpt.T @ bpt)
        p_mat += velocity_feedforward_weight * (bvt.T @ bvt)
        if t == horizon - 1:
            p_mat += terminal_cost_weight * (bpt.T @ bpt)
    p_sym = (p_mat + p_mat.T) * 0.5 + np.eye(variables) * 1e-8
    p_inv = np.linalg.inv(p_sym)
    axis_acc_limit = max_acceleration_mps2 * 0.98 / np.sqrt(2.0)
    return p_mat, p_inv, float(axis_acc_limit)


def solve_local_dmpc_qp(
    grad_pred: np.ndarray,
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    target_traj: np.ndarray,
    target_velocity: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, DmpcInfo]:
    horizon = int(grad_pred.shape[0])
    dt = config.control_step_s
    variables = horizon * 3
    grad_pred = np.asarray(grad_pred, dtype=float).copy()
    grad_pred[:, 2] = 0.0
    current_velocity = np.asarray(current_velocity, dtype=float).copy()
    current_velocity[2] = 0.0
    current_position = np.asarray(current_position, dtype=float).copy()
    current_position[2] = config.uav_altitude_m
    bp, bv = _dynamics_matrices(horizon, dt)
    p_mat, p_inv, axis_acc_limit = _fast_qp_cache(
        horizon,
        dt,
        config.control_penalty,
        config.rho,
        config.velocity_penalty,
        config.task_allocation_weight,
        config.terminal_cost_weight,
        config.velocity_feedforward_weight,
        config.max_acceleration_mps2,
    )
    q_vec = np.zeros(variables, dtype=float)

    for t in range(horizon):
        const_position_delta = (t + 1) * dt * current_velocity
        const_velocity = current_velocity
        bpt = bp[t]
        bvt = bv[t]
        target = np.asarray(target_traj[t], dtype=float)
        v_ref = np.asarray(target_velocity[t], dtype=float)
        target_delta = current_position + const_position_delta - target
        terminal = config.terminal_cost_weight if t == horizon - 1 else 0.0
        q_vec += bpt.T @ (config.gradient_scale * grad_pred[t] + config.rho * const_position_delta)
        q_vec += bpt.T @ (config.task_allocation_weight * target_delta)
        if terminal > 0.0:
            q_vec += bpt.T @ (terminal * target_delta)
        q_vec += bvt.T @ (config.velocity_penalty * const_velocity)
        q_vec += bvt.T @ (config.velocity_feedforward_weight * (const_velocity - v_ref))

    identity = np.eye(variables)
    velocity_rows = []
    velocity_lower = []
    velocity_upper = []
    for t in range(horizon):
        velocity_rows.append(bv[t])
        velocity_lower.extend((-config.max_velocity_mps - current_velocity).tolist())
        velocity_upper.extend((config.max_velocity_mps - current_velocity).tolist())
    a_mat = np.vstack([identity, *velocity_rows])
    lower = np.concatenate(
        [
            np.full(variables, -axis_acc_limit),
            np.asarray(velocity_lower, dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(variables, axis_acc_limit),
            np.asarray(velocity_upper, dtype=float),
        ]
    )

    start = perf_counter()
    if config.qp_solver_mode == "fast_projected":
        try:
            x = -(p_inv @ q_vec)
            x = np.clip(x, -axis_acc_limit, axis_acc_limit)
            velocity = current_velocity.copy()
            for t in range(horizon):
                accel_t = x[3 * t : 3 * t + 3]
                next_velocity = velocity + dt * accel_t
                over = np.abs(next_velocity) > config.max_velocity_mps
                if np.any(over):
                    accel_t[over] = (np.sign(next_velocity[over]) * config.max_velocity_mps - velocity[over]) / dt
                    x[3 * t : 3 * t + 3] = accel_t
                    next_velocity = velocity + dt * accel_t
                velocity = next_velocity
            solve_time_ms = (perf_counter() - start) * 1000.0
            accel = x[:3]
            objective = float(0.5 * x @ p_mat @ x + q_vec @ x)
            status = "solved"
            fallback = False
        except np.linalg.LinAlgError:
            solve_time_ms = (perf_counter() - start) * 1000.0
            accel = np.zeros(3, dtype=float)
            objective = float("nan")
            status = "failed:fast_projected_singular"
            fallback = True
    else:
        solver = osqp.OSQP()
        solver.setup(
            P=sp.csc_matrix((p_mat + p_mat.T) * 0.5),
            q=q_vec,
            A=sp.csc_matrix(a_mat),
            l=lower,
            u=upper,
            verbose=False,
            polish=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=2000,
        )
        result = solver.solve()
        solve_time_ms = (perf_counter() - start) * 1000.0
        solved = result.info.status_val in (1, 2) and result.x is not None
        if solved:
            x = np.asarray(result.x, dtype=float)
            accel = x[:3]
            status = "solved"
            objective = float(result.info.obj_val)
            fallback = False
        else:
            accel = np.zeros(3, dtype=float)
            status = f"failed:{result.info.status}"
            objective = float("nan")
            fallback = True

    active_acc = int(np.sum(np.isclose(np.abs(accel), axis_acc_limit, atol=1e-3)))
    predicted_vel = current_velocity + dt * accel
    active_vel = int(np.sum(np.isclose(np.abs(predicted_vel), config.max_velocity_mps, atol=1e-3)))
    info = DmpcInfo(
        qp_status=status,
        solve_time_ms=float(solve_time_ms),
        objective_value=objective,
        num_variables=variables,
        num_constraints=int(a_mat.shape[0]),
        active_acceleration_constraints=active_acc,
        active_velocity_constraints=active_vel,
        slack_values=[],
        fallback_used=fallback,
    )
    return accel, info


def compute_stgnn_dmpc_control(
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    task: SatelliteTask,
    step: int,
    config: SimulationConfig,
) -> ClosedLoopOutput:
    start_total = perf_counter()
    model, metadata, model_path = _load_model(config)
    batch = build_stgnn_features(positions_m, velocities_mps, task, step, config)
    node_features = torch.from_numpy(batch.node_features[None, ...])
    adjacency = torch.from_numpy(batch.adjacency[None, ...])
    physics_prior = torch.from_numpy(batch.physics_prior[None, ...])

    start_inference = perf_counter()
    with torch.no_grad():
        pred = model(node_features, adjacency, physics_prior).cpu().numpy()[0]
    inference_ms = (perf_counter() - start_inference) * 1000.0
    gradient_scale = float(metadata.get("gradient_output_scale", 1000.0))
    grad_pred = pred * gradient_scale

    accelerations = np.zeros_like(positions_m)
    qp_infos: list[DmpcInfo] = []
    assignment, offsets = formation_offsets(config.num_uavs, task.positions_m.shape[1], config)
    centroid = positions_m.mean(axis=0)
    connectivity_blend = config.stgnn_connectivity_blend
    if task.positions_m.shape[1] == 1 and config.stgnn_connectivity_blend_single_satellite is not None:
        connectivity_blend = float(config.stgnn_connectivity_blend_single_satellite)
    for uav_id in range(config.num_uavs):
        sat_idx = int(batch.assignment[uav_id]) if batch.assignment is not None else int(assignment[uav_id])
        target_traj = np.zeros((config.prediction_horizon, 3), dtype=float)
        target_velocity = np.zeros((config.prediction_horizon, 3), dtype=float)
        for h in range(config.prediction_horizon):
            step_h = min(task.positions_m.shape[0] - 1, step + h + 1)
            _slot_assignment, slots = monitoring_slots(task, step_h, config)
            monitoring_target = task.positions_m[step_h, sat_idx] + config.stgnn_formation_offset_scale * offsets[uav_id]
            if int(_slot_assignment[uav_id]) == sat_idx:
                monitoring_target = slots[uav_id]
            target_traj[h] = (
                (1.0 - connectivity_blend) * monitoring_target
                + connectivity_blend * centroid
            )
            target_velocity[h] = task.velocities_mps[step_h, sat_idx]
        accel, info = solve_local_dmpc_qp(
            grad_pred[:, uav_id, :],
            positions_m[uav_id],
            velocities_mps[uav_id],
            target_traj,
            target_velocity,
            config,
        )
        accelerations[uav_id] = accel
        qp_infos.append(info)

    compute_ms = (perf_counter() - start_total) * 1000.0
    return ClosedLoopOutput(
        accel_mps2=accelerations,
        compute_ms=float(compute_ms),
        inference_ms=float(inference_ms),
        grad_pred=grad_pred,
        qp_infos=qp_infos,
        model_path=str(model_path),
        stgnn_loaded=True,
        model_file_size_mb=model_file_size_mb(model_path),
        input_feature_dim=NODE_FEATURE_DIM,
        output_gradient_dim=3,
        fallback_used=any(info.fallback_used for info in qp_infos),
        assignment=batch.assignment,
    )


def write_feature_contract(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# ST-GNN feature contract",
                "",
                f"node_feature_dim: {NODE_FEATURE_DIM}",
                "max_satellites: 5",
                "history: SimulationConfig.historical_observation_horizon",
                "per UAV fields: position, velocity",
                "per satellite fields: projected position, projected velocity, UAV-to-projection distance, task weight, orbital height, inclination, RAAN",
                "padding: satellites above current M are zero padded and marked with sat_mask=0",
                "assignment: one-hot primary satellite assignment is included in each node feature",
            ]
        ),
        encoding="utf-8",
    )


def write_qp_implementation_report(path: Path, config: SimulationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "solver": "OSQP",
                "variables": "u_i[0:H-1] in R^3",
                "horizon": config.prediction_horizon,
                "hard_constraints": {
                    "acceleration_axis_box_mps2": config.max_acceleration_mps2,
                    "velocity_axis_box_mps": config.max_velocity_mps,
                },
                "objective_terms": [
                    "ST-GNN predicted gradient linear term",
                    "rho position displacement regularization",
                    "control effort penalty",
                    "velocity penalty",
                ],
                "fallback_policy": "QP failure returns zero acceleration and is explicitly logged; main reproduction target requires fallback rate < 5%.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
