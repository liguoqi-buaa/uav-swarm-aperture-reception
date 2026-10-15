from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .config import SimulationConfig
from .environment import SatelliteTask
from .potential import composite_potential


@dataclass(frozen=True, slots=True)
class AlgorithmProfile:
    name: str
    gradient_gain: float
    attr_scale: float
    rep_scale: float
    conn_scale: float
    power_scale: float
    prediction_steps: int
    prediction_noise: float
    response_delay_steps: int
    compute_cost_ms: float
    inference_cost_ms: float | None


@dataclass(slots=True)
class ControlCommand:
    accel_mps2: np.ndarray
    compute_ms: float
    inference_ms: float | None
    grad_pred: np.ndarray | None
    qp_status: list[str]
    qp_solve_time_ms: list[float]
    active_constraints: list[str]
    fallback_used: bool
    stgnn_loaded: bool
    model_path: str | None
    model_file_size_mb: float | None
    input_feature_dim: int | None
    output_gradient_dim: int | None
    assignment: np.ndarray | None


ALGORITHM_PROFILES: dict[str, AlgorithmProfile] = {
    "OLSR-FANET": AlgorithmProfile("OLSR-FANET", 0.05, 0.12, 0.35, 1.25, 0.8, 0, 0.22, 10, 85.0, None),
    "APF-TC": AlgorithmProfile("APF-TC", 0.11, 0.32, 1.0, 0.65, 0.7, 0, 0.14, 6, 68.0, None),
    "GNN-TC": AlgorithmProfile("GNN-TC", 0.12, 0.38, 0.75, 0.75, 0.55, 1, 0.10, 4, 45.0, 28.0),
    "DMPC-C": AlgorithmProfile("DMPC-C", 0.14, 0.42, 0.9, 0.9, 0.55, 0, 0.08, 3, 156.0, None),
    "ST-GNN-DMPC": AlgorithmProfile("ST-GNN-DMPC", 0.92, 1.15, 1.0, 1.25, 0.45, 5, 0.02, 1, 38.0, 12.0),
}

# 公平性对照（profile_gain_mode="matched"）：把全部基线拉齐到与 ST-GNN-DMPC 相同的
# 势场增益与零预测噪声，只保留「预测步数 prediction_steps」与「响应时延
# response_delay_steps」两个差异。这样才可以把 ST-GNN-DMPC 的优势来源从
# 「增益调参」隔离成「预测能力」。
MATCHED_GAIN_REFERENCE = "ST-GNN-DMPC"


def matched_profile(algorithm: str) -> AlgorithmProfile:
    base = ALGORITHM_PROFILES[algorithm]
    if algorithm == MATCHED_GAIN_REFERENCE:
        return base
    ref = ALGORITHM_PROFILES[MATCHED_GAIN_REFERENCE]
    return replace(
        base,
        gradient_gain=ref.gradient_gain,
        attr_scale=ref.attr_scale,
        rep_scale=ref.rep_scale,
        conn_scale=ref.conn_scale,
        power_scale=ref.power_scale,
        prediction_noise=0.0,
    )


def profile_for(algorithm: str, config: SimulationConfig) -> AlgorithmProfile:
    if getattr(config, "profile_gain_mode", "published") == "matched":
        return matched_profile(algorithm)
    return ALGORITHM_PROFILES[algorithm]


def _future_gradient(
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    task: SatelliteTask,
    step: int,
    config: SimulationConfig,
    profile: AlgorithmProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    max_step = task.positions_m.shape[0] - 1
    horizon = max(0, min(profile.prediction_steps, config.prediction_horizon))
    steps = [min(max_step, step + h) for h in range(horizon + 1)]
    gradients = []
    for h_step in steps:
        result = composite_potential(
            positions_m,
            velocities_mps,
            task,
            h_step,
            config,
            gains_scale=(profile.attr_scale, profile.rep_scale, profile.conn_scale, profile.power_scale),
        )
        gradients.append(result.gradient)
    grad = np.mean(gradients, axis=0)
    if profile.prediction_noise > 0.0:
        scale = np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1.0)
        grad = grad + rng.normal(0.0, profile.prediction_noise, size=grad.shape) * scale
    return grad


def compute_control(
    algorithm: str,
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    task: SatelliteTask,
    step: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    geometry: dict | None = None,
) -> ControlCommand:
    from .independent_baselines import independent_control, parse_independent

    spec = parse_independent(algorithm)
    if spec is not None:
        if geometry is None:
            raise ValueError(f"{algorithm} 需要 geometry（scenario C 孔径几何）")
        start = perf_counter()
        accel = independent_control(spec, positions_m, velocities_mps, step, config, geometry)
        measured_ms = (perf_counter() - start) * 1000.0
        return ControlCommand(
            accel_mps2=accel,
            compute_ms=measured_ms,
            inference_ms=None,
            grad_pred=None,
            qp_status=[f"independent:{spec.mode}:{spec.kind}"] * positions_m.shape[0],
            qp_solve_time_ms=[],
            active_constraints=[],
            fallback_used=False,
            stgnn_loaded=False,
            model_path=None,
            model_file_size_mb=None,
            input_feature_dim=None,
            output_gradient_dim=None,
            assignment=None,
        )
    if algorithm == "ST-GNN-DMPC":
        from src.controllers.stgnn_dmpc_controller import compute_stgnn_dmpc_control

        output = compute_stgnn_dmpc_control(positions_m, velocities_mps, task, step, config)
        active = [
            f"acc:{info.active_acceleration_constraints};vel:{info.active_velocity_constraints}"
            for info in output.qp_infos
        ]
        return ControlCommand(
            accel_mps2=output.accel_mps2,
            compute_ms=output.compute_ms,
            inference_ms=output.inference_ms,
            grad_pred=output.grad_pred,
            qp_status=[info.qp_status for info in output.qp_infos],
            qp_solve_time_ms=[info.solve_time_ms for info in output.qp_infos],
            active_constraints=active,
            fallback_used=output.fallback_used,
            stgnn_loaded=output.stgnn_loaded,
            model_path=output.model_path,
            model_file_size_mb=output.model_file_size_mb,
            input_feature_dim=output.input_feature_dim,
            output_gradient_dim=output.output_gradient_dim,
            assignment=output.assignment,
        )
    profile = profile_for(algorithm, config)
    start = perf_counter()
    effective_step = max(0, step - profile.response_delay_steps)
    grad = _future_gradient(positions_m, velocities_mps, task, effective_step, config, profile, rng)
    delta = -profile.gradient_gain * grad / max(config.rho, 1e-6)
    accel = 2.0 * (delta - velocities_mps * config.control_step_s) / (config.control_step_s**2)
    norms = np.linalg.norm(accel, axis=1, keepdims=True)
    accel = np.where(norms > config.max_acceleration_mps2, accel / np.maximum(norms, 1e-9) * config.max_acceleration_mps2, accel)
    accel[:, 2] = np.clip(accel[:, 2], -0.5, 0.5)
    measured_ms = (perf_counter() - start) * 1000.0
    # 诚实计时：直接上报实测墙钟。原实现为 `max(measured_ms, profile.compute_cost_ms)`，
    # 即把硬编码常数当作算力下限，导致 baseline 的 runtime 方差恒为 0（非实测）。
    compute_ms = measured_ms
    return ControlCommand(
        accel_mps2=accel,
        compute_ms=compute_ms,
        inference_ms=profile.inference_cost_ms,
        grad_pred=grad,
        qp_status=["profile_controller"] * positions_m.shape[0],
        qp_solve_time_ms=[],
        active_constraints=[],
        fallback_used=False,
        stgnn_loaded=False,
        model_path=None,
        model_file_size_mb=None,
        input_feature_dim=None,
        output_gradient_dim=None,
        assignment=None,
    )


def step_dynamics(
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    accel_mps2: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dt = config.control_step_s
    new_positions = positions_m + velocities_mps * dt + 0.5 * accel_mps2 * dt * dt
    new_velocities = velocities_mps + accel_mps2 * dt
    speed = np.linalg.norm(new_velocities, axis=1, keepdims=True)
    new_velocities = np.where(speed > config.max_velocity_mps, new_velocities / np.maximum(speed, 1e-9) * config.max_velocity_mps, new_velocities)
    new_positions[:, 2] = config.uav_altitude_m
    new_velocities[:, 2] = 0.0
    return new_positions, new_velocities
