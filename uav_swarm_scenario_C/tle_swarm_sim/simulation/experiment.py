from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tle_swarm_sim.sgp4_utils import ecef_to_geodetic, satellite_ecef_at
from tle_swarm_sim.tle import TleRecord, load_tle_records

from .config import SimulationConfig
from .controllers import ALGORITHM_PROFILES, ControlCommand, compute_control, step_dynamics
from .environment import SatelliteTask, build_satellite_task, initial_swarm_state
from .network import adjacency_matrix, algebraic_connectivity, average_node_degree, link_disruption_rate
from .potential import monitoring_performance, tracking_error
from .reporting import generate_reports


@dataclass(frozen=True, slots=True)
class RunResult:
    algorithm: str
    scenario: str
    m: int
    trial: int
    random_seed: int
    sample_id: str
    result_path: str
    raw: dict[str, float]


def _load_records(config: SimulationConfig) -> list[TleRecord]:
    archive = config.processed_dir / "starlink_archive.tle"
    if not archive.exists():
        raise FileNotFoundError(f"Missing processed archive: {archive}")
    return load_tle_records(archive)


def _load_sample_ids(config: SimulationConfig) -> list[str]:
    path = config.processed_dir / "all_samples.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [str(row.get("sample_id", "")) for row in rows]


def _select_group(records: list[TleRecord], group_size: int, trial: int, rng: np.random.Generator) -> list[TleRecord]:
    ordered = sorted(records, key=lambda r: (r.epoch_datetime, int(r.catalog_number)))
    if len(ordered) < group_size:
        return list(rng.choice(ordered, size=group_size, replace=True))
    start = (trial * group_size) % len(ordered)
    group = []
    seen = set()
    idx = start
    while len(group) < group_size and len(seen) < len(ordered):
        record = ordered[idx % len(ordered)]
        idx += 1
        if record.catalog_number in seen:
            continue
        seen.add(record.catalog_number)
        group.append(record)
    if len(group) < group_size:
        group = list(rng.choice(ordered, size=group_size, replace=False))
    return group


def _scenario_label(group_size: int) -> str:
    return "single_monitoring" if group_size == 1 else f"multi{group_size}_monitoring"


def _portable_result_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _trial_seed(config: SimulationConfig, scenario_index: int, algorithm_index: int, trial: int) -> int:
    _ = algorithm_index
    return int(config.random_seed + scenario_index * 100_000 + trial)


def _compact_adjacency(adj: np.ndarray) -> str:
    return ";".join("".join(str(int(v)) for v in row) for row in adj)


def _extended_window_lookup(config: SimulationConfig) -> dict[str, dict]:
    path = config.task_window_search_csv
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "elevation_threshold_deg" in df.columns:
        df = df[np.isclose(df["elevation_threshold_deg"].astype(float), config.task_window_main_elevation_deg)]
    rows: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        catalog = str(row.get("catalog_number", ""))
        if catalog and catalog not in rows:
            rows[catalog] = row
    return rows


def _select_task_start_time(group: list[TleRecord], config: SimulationConfig) -> datetime:
    if getattr(config, "task_window_mode", "phase4_local") != "extended_pass_10deg":
        return group[0].epoch_datetime.astimezone(timezone.utc)
    lookup = _extended_window_lookup(config)
    row = lookup.get(str(group[0].catalog_number))
    if not row:
        return group[0].epoch_datetime.astimezone(timezone.utc)
    aos = str(row.get("AOS_utc", ""))
    if not aos:
        return group[0].epoch_datetime.astimezone(timezone.utc)
    return datetime.fromisoformat(aos).astimezone(timezone.utc)


def _estimate_convergence_latency_ms(
    tracking_values: list[float],
    perf_values: list[float],
    lambda_values: list[float],
    config: SimulationConfig,
) -> float:
    if len(tracking_values) < 5:
        return float(len(tracking_values) * config.control_step_s * 1000.0)
    err = np.asarray(tracking_values, dtype=float)
    perf = np.asarray(perf_values, dtype=float)
    lamb = np.asarray(lambda_values, dtype=float)
    final_err = float(np.median(err[-5:]))
    initial_err = float(err[0])
    target_err = final_err + 0.05 * max(0.0, initial_err - final_err)
    final_perf = float(np.median(perf[-5:]))
    for idx in range(2, len(err)):
        if (
            np.all(err[max(0, idx - 2) : idx + 1] <= target_err)
            and np.all(perf[max(0, idx - 2) : idx + 1] >= 0.95 * final_perf)
            and np.all(lamb[max(0, idx - 2) : idx + 1] >= 0.0)
        ):
            return float(idx * config.control_step_s * 1000.0)

    residual = err - final_err
    valid = residual > max(1.0, 0.01 * max(initial_err, 1.0))
    if valid.sum() >= 4 and initial_err > final_err:
        t = np.arange(len(err), dtype=float)[valid] * config.control_step_s
        y = np.log(residual[valid])
        slope, _intercept = np.polyfit(t, y, 1)
        if slope < -1e-6:
            return float(min(len(err) * config.control_step_s * 1000.0, -1000.0 / slope))
    return float(len(err) * config.control_step_s * 1000.0)


def _estimate_topology_response_latency_ms(
    algorithm: str,
    compute_times: list[float],
    config: SimulationConfig,
) -> float:
    profile = ALGORITHM_PROFILES.get(algorithm)
    decision_time = float(np.median(compute_times)) if compute_times else (
        profile.compute_cost_ms if profile is not None else 1.0)
    delay_steps = profile.response_delay_steps if profile is not None else 0
    return float(delay_steps * config.sample_time_s * 1000.0 + decision_time)


def _write_trial_artifacts(
    trial_dir: Path,
    config: SimulationConfig,
    group: list[TleRecord],
    result: RunResult,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    uav_rows: list[dict],
    sat_rows: list[dict],
    metric_rows: list[dict],
    runtime: dict,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "algorithm": result.algorithm,
                "scenario": result.scenario,
                "M": result.m,
                "trial_id": result.trial,
                "random_seed": result.random_seed,
                "num_uavs": config.num_uavs,
                "duration_s": config.simulation_duration_s,
                "control_step_s": config.control_step_s,
                "monitoring_threshold": config.monitoring_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (trial_dir / "satellite_sample_ids.json").write_text(
        json.dumps(
            {
                "sample_id": result.sample_id,
                "satellites": [
                    {
                        "name": record.name,
                        "catalog_number": record.catalog_number,
                        "epoch": record.epoch_datetime.isoformat(),
                    }
                    for record in group
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "uav_id": idx,
                "x_m": initial_positions[idx, 0],
                "y_m": initial_positions[idx, 1],
                "z_m": initial_positions[idx, 2],
                "vx_mps": initial_velocities[idx, 0],
                "vy_mps": initial_velocities[idx, 1],
                "vz_mps": initial_velocities[idx, 2],
            }
            for idx in range(initial_positions.shape[0])
        ]
    ).to_csv(trial_dir / "uav_initial_states.csv", index=False)
    pd.DataFrame(uav_rows).to_csv(trial_dir / "trajectory_uav.csv", index=False)
    pd.DataFrame(sat_rows).to_csv(trial_dir / "trajectory_satellite.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(trial_dir / "metrics_timeseries.csv", index=False)
    (trial_dir / "metrics_summary.json").write_text(json.dumps(result.raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (trial_dir / "runtime_log.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")


def _control_trace_rows(
    scenario_name: str,
    trial: int,
    step: int,
    algorithm: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    command: ControlCommand,
) -> list[dict]:
    rows = []
    accel = command.accel_mps2
    for uav_id in range(positions.shape[0]):
        grad = command.grad_pred[0, uav_id] if command.grad_pred is not None and command.grad_pred.ndim == 3 else np.full(3, np.nan)
        rows.append(
            {
                "scenario": scenario_name,
                "trial_id": trial,
                "time_step": step,
                "algorithm": algorithm,
                "uav_id": uav_id,
                "grad_x": float(grad[0]),
                "grad_y": float(grad[1]),
                "grad_z": float(grad[2]),
                "control_x": float(accel[uav_id, 0]),
                "control_y": float(accel[uav_id, 1]),
                "control_z": float(accel[uav_id, 2]),
                "control_norm": float(np.linalg.norm(accel[uav_id])),
                "position_x": float(positions[uav_id, 0]),
                "position_y": float(positions[uav_id, 1]),
                "position_z": float(positions[uav_id, 2]),
                "velocity_norm": float(np.linalg.norm(velocities[uav_id])),
                "active_constraints": command.active_constraints[uav_id] if uav_id < len(command.active_constraints) else "",
                "qp_status": command.qp_status[uav_id] if uav_id < len(command.qp_status) else "",
                "fallback_used": bool(command.fallback_used),
                "stgnn_loaded": bool(command.stgnn_loaded),
            }
        )
    return rows


def _run_single_trial(
    algorithm: str,
    records: list[TleRecord],
    sample_ids: list[str],
    scenario_name: str,
    group_size: int,
    scenario_index: int,
    algorithm_index: int,
    trial: int,
    config: SimulationConfig,
    export_trials: bool = True,
) -> tuple[RunResult, list[dict], dict[str, list[dict]]]:
    seed = _trial_seed(config, scenario_index, algorithm_index, trial)
    rng = np.random.default_rng(seed)
    aperture_geo = None
    if getattr(config, "scenario_mode", "legacy") == "aperture":
        # Scenario C: real-geometry fixed monitoring area + co-visible satellites.
        from .scenario_c import (
            aperture_quality,
            build_aperture_task,
            group_records_for_window,
            select_aperture_window,
        )

        window = select_aperture_window(records, config, trial)
        group = group_records_for_window(records, window)
        group_size = len(group)
        start_time = datetime.fromisoformat(window["start_utc"])
        task, aperture_geo = build_aperture_task(group, start_time, config)
    else:
        group = _select_group(records, group_size, trial, rng)
        start_time = _select_task_start_time(group, config)
        site_ecef = satellite_ecef_at(group[0], start_time)
        site_lat, site_lon, _ = ecef_to_geodetic(site_ecef)
        task = build_satellite_task(group, start_time, site_lat, site_lon, config)
    positions, velocities = initial_swarm_state(task, config, rng)
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()

    perf_values: list[float] = []
    tracking_values: list[float] = []
    lambda_values: list[float] = []
    degree_values: list[float] = []
    compute_times: list[float] = []
    inference_times: list[float] = []
    disruption_values: list[float] = []
    connected_graph_values: list[float] = []
    link_availability_values: list[float] = []
    k_connectivity_values: list[float] = []
    metric_rows: list[dict] = []
    uav_rows: list[dict] = []
    sat_rows: list[dict] = []
    ns3_rows: list[dict] = []
    control_trace: list[dict] = []
    qp_rows: list[dict] = []
    allocation_rows: list[dict] = []
    fallback_steps = 0
    qp_solved = 0
    qp_total = 0
    aperture_rows: list[dict] = []
    acceleration_clip_hits = 0
    acceleration_total = 0
    stgnn_loaded = False
    model_path = None
    model_file_size = None
    input_feature_dim = None
    output_gradient_dim = None
    previous_adj = None
    converged_step = None
    stable_count = 0

    steps = task.positions_m.shape[0]
    for step in range(steps):
        command = compute_control(algorithm, positions, velocities, task, step, config, rng, geometry=aperture_geo)
        accel, compute_ms, inference_ms = command.accel_mps2, command.compute_ms, command.inference_ms
        if algorithm == "ST-GNN-DMPC":
            stgnn_loaded = stgnn_loaded or command.stgnn_loaded
            model_path = command.model_path or model_path
            model_file_size = command.model_file_size_mb or model_file_size
            input_feature_dim = command.input_feature_dim or input_feature_dim
            output_gradient_dim = command.output_gradient_dim or output_gradient_dim
            fallback_steps += int(command.fallback_used)
            qp_solved += sum(1 for status in command.qp_status if status == "solved")
            qp_total += len(command.qp_status)
            norms_before_step = np.linalg.norm(accel, axis=1)
            acceleration_clip_hits += int(np.sum(norms_before_step >= config.max_acceleration_mps2 - 1e-3))
            acceleration_total += accel.shape[0]
            if step < 10:
                control_trace.extend(_control_trace_rows(scenario_name, trial, step, algorithm, positions, velocities, command))
            for uav_id, status in enumerate(command.qp_status):
                qp_rows.append(
                    {
                        "scenario": scenario_name,
                        "M": group_size,
                        "trial_id": trial,
                        "step": step,
                        "uav_id": uav_id,
                        "qp_status": status,
                        "solve_time_ms": command.qp_solve_time_ms[uav_id] if uav_id < len(command.qp_solve_time_ms) else np.nan,
                        "active_constraints": command.active_constraints[uav_id] if uav_id < len(command.active_constraints) else "",
                        "fallback_used": command.fallback_used,
                    }
                )
            if command.assignment is not None:
                for uav_id, sat_id in enumerate(command.assignment):
                    allocation_rows.append(
                        {
                            "scenario": scenario_name,
                            "M": group_size,
                            "trial_id": trial,
                            "step": step,
                            "uav_id": uav_id,
                            "primary_satellite": int(sat_id),
                        }
                    )
        positions, velocities = step_dynamics(positions, velocities, accel, config)
        adj = adjacency_matrix(positions, config, previous_adj)
        lambda2 = algebraic_connectivity(adj)
        disruption = link_disruption_rate(previous_adj, adj)
        degree = average_node_degree(adj)
        connected_links = int(np.triu(adj, k=1).sum())
        possible_links = int(config.num_uavs * (config.num_uavs - 1) / 2)
        connected_graph = float(lambda2 > 1e-9)
        link_availability = float(connected_links / possible_links) if possible_links else 0.0
        k_satisfied = float(lambda2 >= config.lambda_threshold)

        perf = monitoring_performance(positions, task, step, config)
        err = tracking_error(positions, task, step, config)

        aperture = aperture_quality(positions[:, :2], aperture_geo, step, config) if aperture_geo is not None else None
        if aperture is not None:
            aperture_rows.append(aperture)

        perf_values.append(perf)
        tracking_values.append(err)
        lambda_values.append(lambda2)
        degree_values.append(degree)
        disruption_values.append(disruption)
        connected_graph_values.append(connected_graph)
        link_availability_values.append(link_availability)
        k_connectivity_values.append(k_satisfied)
        compute_times.append(compute_ms)
        if inference_ms is not None:
            inference_times.append(inference_ms)

        row = {
            "step": step,
            "time_s": step * config.control_step_s,
            "monitoring_performance": perf,
            "effective": int(perf >= config.monitoring_threshold),
            "tracking_error_m": err,
            "algebraic_connectivity": lambda2,
            "link_disruption_rate": disruption,
            "average_node_degree": degree,
            "connected_links": connected_links,
            "possible_links": possible_links,
            "connected_graph_ratio": connected_graph,
            "link_availability_ratio": link_availability,
            "k_connectivity_satisfaction_ratio": k_satisfied,
            "average_single_step_computational_time_ms": compute_ms,
            "inference_latency_ms": inference_ms if inference_ms is not None else np.nan,
        }
        if aperture is not None:
            row.update(aperture)
        metric_rows.append(row)
        if group_size in (1, 3):
            ns3_rows.append(
                {
                    "scenario": scenario_name,
                    "M": group_size,
                    "algorithm": algorithm,
                    "trial_id": trial,
                    "random_seed": seed,
                    **row,
                    "adjacency_matrix": _compact_adjacency(adj),
                }
            )
        if export_trials:
            for uav_id in range(config.num_uavs):
                uav_rows.append(
                    {
                        "step": step,
                        "time_s": step * config.control_step_s,
                        "uav_id": uav_id,
                        "x_m": positions[uav_id, 0],
                        "y_m": positions[uav_id, 1],
                        "z_m": positions[uav_id, 2],
                        "vx_mps": velocities[uav_id, 0],
                        "vy_mps": velocities[uav_id, 1],
                        "vz_mps": velocities[uav_id, 2],
                    }
                )
            for sat_id in range(group_size):
                if aperture_geo is not None:
                    # Scenario C：导出真实星下点地面轨迹（相对监测区中心，米）
                    sx, sy = float(aperture_geo["sub_xy"][step, sat_id, 0]), float(aperture_geo["sub_xy"][step, sat_id, 1])
                    svx = float((aperture_geo["sub_xy"][min(step + 1, steps - 1), sat_id, 0]
                                 - aperture_geo["sub_xy"][max(step - 1, 0), sat_id, 0]) / (2.0 * config.control_step_s))
                    svy = float((aperture_geo["sub_xy"][min(step + 1, steps - 1), sat_id, 1]
                                 - aperture_geo["sub_xy"][max(step - 1, 0), sat_id, 1]) / (2.0 * config.control_step_s))
                else:
                    sx, sy = task.positions_m[step, sat_id, 0], task.positions_m[step, sat_id, 1]
                    svx = task.velocities_mps[step, sat_id, 0]
                    svy = task.velocities_mps[step, sat_id, 1]
                sat_rows.append(
                    {
                        "step": step,
                        "time_s": step * config.control_step_s,
                        "satellite_group_index": sat_id,
                        "catalog_number": group[sat_id].catalog_number,
                        "x_m": sx,
                        "y_m": sy,
                        "z_m": float(config.uav_altitude_m),
                        "vx_mps": svx,
                        "vy_mps": svy,
                        "vz_mps": 0.0,
                        "subpoint_distance_m": float(aperture_geo["sub_dist_m"][step, sat_id]) if aperture_geo is not None else float("nan"),
                        "elevation_deg": float(aperture_geo["elev_deg"][step, sat_id]) if aperture_geo is not None else float("nan"),
                    }
                )

        previous_adj = adj
        stable = np.linalg.norm(accel, axis=1).mean() < 0.25 and lambda2 >= config.lambda_threshold
        stable_count = stable_count + 1 if stable else 0
        if converged_step is None and stable_count >= 3:
            converged_step = step

    sample_id = sample_ids[(trial + scenario_index * config.trials) % len(sample_ids)] if sample_ids else f"{scenario_name}-{trial:03d}"
    raw = {
        "monitoring_performance": float(np.mean(perf_values)),
        "effective_monitoring_ratio": float(100.0 * np.mean(np.array(perf_values) >= config.monitoring_threshold)),
        "tracking_error": float(np.mean(tracking_values)),
        "average_connectivity": float(np.mean(lambda_values)),
        "link_disruption_rate": float(100.0 * np.mean(disruption_values)),
        "average_node_degree": float(np.mean(degree_values)),
        "connected_graph_ratio": float(100.0 * np.mean(connected_graph_values)),
        "link_availability_ratio": float(100.0 * np.mean(link_availability_values)),
        "k_connectivity_satisfaction_ratio": float(100.0 * np.mean(k_connectivity_values)),
        "topology_convergence_latency_ms": (
            float(converged_step * config.control_step_s * 1000.0)
            if converged_step is not None
            else float(steps * config.control_step_s * 1000.0)
        ),
        "topology_convergence_latency_legacy_ms": _estimate_topology_response_latency_ms(algorithm, compute_times, config),
        "average_single_step_ms": float(np.mean(compute_times)),
        "inference_latency_ms": float(np.mean(inference_times)) if inference_times else float("nan"),
        "stgnn_loaded": float(stgnn_loaded),
        "qp_solved_ratio": float(qp_solved / qp_total) if qp_total else float("nan"),
        "fallback_rate": float(fallback_steps / steps) if algorithm == "ST-GNN-DMPC" else 0.0,
        "acceleration_clip_ratio": float(acceleration_clip_hits / acceleration_total) if acceleration_total else float("nan"),
    }
    if aperture_geo is not None:
        from .scenario_c import aperture_summary

        raw.update(aperture_summary(aperture_rows, aperture_geo, config))
        # 孔径场景下 monitoring_threshold(0.79) 是旧缩放域的阈值，直接套用会恒为 0；
        # 改用「相对同一步最优孔径的达成率」这一尺度无关判据。
        _gains = np.asarray([row["aperture_gain"] for row in aperture_rows], dtype=float)
        _oracle = np.asarray(aperture_geo["oracle_gain_step"], dtype=float)
        _ok = np.isfinite(_gains) & np.isfinite(_oracle) & (_oracle > 0.0)
        raw["effective_monitoring_ratio"] = (
            float(100.0 * np.mean(_gains[_ok] >= 0.5 * _oracle[_ok])) if bool(_ok.any()) else float("nan")
        )
    _tag = getattr(config, "report_tag", "") or ""
    _tag = f"_{_tag}" if _tag else ""
    result_path = config.reports_dir / "trials" / f"M{group_size}" / algorithm / f"{scenario_name}{_tag}_trial_{trial:02d}"
    result = RunResult(
        algorithm=algorithm,
        scenario=scenario_name,
        m=group_size,
        trial=trial,
        random_seed=seed,
        sample_id=sample_id,
        result_path=_portable_result_path(result_path, config.repo_root),
        raw=raw,
    )
    if export_trials:
        _write_trial_artifacts(
            result_path,
            config,
            group,
            result,
            initial_positions,
            initial_velocities,
            uav_rows,
            sat_rows,
            metric_rows,
            runtime={
                "compute_time_model": (
                    "stgnn_osqp_wall_clock" if algorithm == "ST-GNN-DMPC"
                    else "independent_pd_wall_clock" if algorithm.startswith(("B1_", "B2_"))
                    else "profile_wall_clock"
                ),
                "steps": steps,
                "control_step_s": config.control_step_s,
                "model_path": model_path,
                "stgnn_loaded": stgnn_loaded,
                "model_file_size_mb": model_file_size,
                "input_feature_dim": input_feature_dim,
                "output_gradient_dim": output_gradient_dim,
                "fallback_used": bool(fallback_steps > 0),
                "fallback_rate": raw["fallback_rate"],
                "qp_solved_ratio": raw["qp_solved_ratio"],
                "acceleration_clip_ratio": raw["acceleration_clip_ratio"],
            },
        )
    return result, ns3_rows, {"control_trace": control_trace, "qp_rows": qp_rows, "allocation_rows": allocation_rows}


def _summarize_trials(results: list[RunResult], metric: str, scenario_key: str) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHM_PROFILES:
        series = [r.raw[metric] for r in results if r.algorithm == algorithm and r.scenario == scenario_key]
        rows.append(
            {
                "algorithm": algorithm,
                "mean": float(np.mean(series)) if series else float("nan"),
                "std": float(np.std(series, ddof=1)) if len(series) > 1 else float("nan"),
                "num_trials": len(series),
            }
        )
    return pd.DataFrame(rows)


def _write_raw_trial_tables(results: list[RunResult], config: SimulationConfig) -> None:
    raw_dir = config.reports_dir / "raw_trials"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_rows = []
    for result in results:
        base = {
            "scenario": result.scenario,
            "M": result.m,
            "algorithm": result.algorithm,
            "trial_id": result.trial,
            "random_seed": result.random_seed,
            "sample_id": result.sample_id,
            "result_path": result.result_path,
        }
        base_rows.append({**base, **result.raw})
    df = pd.DataFrame(base_rows)
    df[df["scenario"] == "single_monitoring"].to_csv(raw_dir / "table2_single_monitoring_trials.csv", index=False)
    df[df["scenario"] == "multi3_monitoring"].to_csv(raw_dir / "table3_multi3_monitoring_trials.csv", index=False)
    df[df["scenario"].isin(["single_monitoring", "multi3_monitoring"])].to_csv(raw_dir / "table4_network_trials.csv", index=False)
    df[df["scenario"] == "realtime"].to_csv(raw_dir / "table5_realtime_trials.csv", index=False)
    trials_dir = config.reports_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    df.rename(
        columns={
            "monitoring_performance": "average_monitoring_performance",
            "effective_monitoring_ratio": "effective_monitoring_duration_ratio",
            "tracking_error": "average_tracking_error",
            "average_connectivity": "average_network_connectivity",
            "average_single_step_ms": "average_single_step_computational_time_ms",
        }
    ).to_csv(trials_dir / "trial_index.csv", index=False)


def _write_ns3_dynamic(ns3_rows: list[dict], config: SimulationConfig) -> None:
    ns3_dir = config.reports_dir / "ns3"
    ns3_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ns3_rows)
    if df.empty:
        return
    df.to_csv(ns3_dir / "dynamic_link_log.csv", index=False)
    metrics_cols = [
        "scenario",
        "M",
        "algorithm",
        "trial_id",
        "random_seed",
        "step",
        "time_s",
        "algebraic_connectivity",
        "link_disruption_rate",
        "average_node_degree",
        "connected_links",
        "possible_links",
        "connected_graph_ratio",
        "link_availability_ratio",
        "k_connectivity_satisfaction_ratio",
    ]
    df[metrics_cols].to_csv(ns3_dir / "network_metrics_timeseries.csv", index=False)
    summary = (
        df.groupby(["scenario", "M", "algorithm"])
        .agg(
            algebraic_connectivity_mean=("algebraic_connectivity", "mean"),
            algebraic_connectivity_std=("algebraic_connectivity", "std"),
            link_disruption_rate_pct_mean=("link_disruption_rate", lambda s: 100.0 * float(np.mean(s))),
            link_disruption_rate_pct_std=("link_disruption_rate", lambda s: 100.0 * float(np.std(s, ddof=1))),
            average_node_degree_mean=("average_node_degree", "mean"),
            average_node_degree_std=("average_node_degree", "std"),
            connected_graph_ratio_pct_mean=("connected_graph_ratio", lambda s: 100.0 * float(np.mean(s))),
            link_availability_ratio_pct_mean=("link_availability_ratio", lambda s: 100.0 * float(np.mean(s))),
            k_connectivity_satisfaction_ratio_pct_mean=("k_connectivity_satisfaction_ratio", lambda s: 100.0 * float(np.mean(s))),
            connected_links_mean=("connected_links", "mean"),
        )
        .reset_index()
    )
    (ns3_dir / "network_metrics_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    (ns3_dir / "ns3_config_used.json").write_text(
        json.dumps(
            {
                "carrier_frequency_hz": config.carrier_frequency_hz,
                "transmit_power_dbm": config.transmit_power_dbm,
                "bandwidth_hz": config.bandwidth_hz,
                "connectivity_snr_threshold_db": config.connectivity_snr_threshold_db,
                "max_communication_range_m": config.max_communication_range_m,
                "model_note": "Dynamic link states computed from the same 3GPP-style SNR formula used by the simulation at every control step.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_closed_loop_debug(debug_rows: dict[str, list[dict]], config: SimulationConfig) -> None:
    debug_dir = config.reports_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    control_df = pd.DataFrame(debug_rows.get("control_trace", []))
    qp_df = pd.DataFrame(debug_rows.get("qp_rows", []))
    allocation_df = pd.DataFrame(debug_rows.get("allocation_rows", []))
    if not control_df.empty:
        control_df.to_csv(debug_dir / "stgnn_closed_loop_trace.csv", index=False)
        control_df.to_csv(debug_dir / "control_input_comparison.csv", index=False)
    if not qp_df.empty:
        summary = (
            qp_df.groupby(["scenario", "M", "trial_id", "qp_status"])
            .size()
            .reset_index(name="count")
        )
        summary.to_csv(debug_dir / "qp_solver_status_summary.csv", index=False)
        qp_df[qp_df["qp_status"] != "solved"].to_csv(debug_dir / "qp_solver_failure_cases.csv", index=False)
    if not allocation_df.empty:
        allocation_df.to_csv(debug_dir / "task_allocation_runtime_trace.csv", index=False)
        allocation_df[allocation_df["M"] == 3].to_csv(debug_dir / "uav_satellite_assignment_M3.csv", index=False)
        allocation_df[allocation_df["M"] == 5].to_csv(debug_dir / "uav_satellite_assignment_M5.csv", index=False)
        counts = allocation_df.groupby(["scenario", "M", "primary_satellite"]).size().reset_index(name="assigned_steps")
        (debug_dir / "task_allocation_effect_summary.md").write_text(counts.to_markdown(index=False), encoding="utf-8")
    (debug_dir / "stgnn_gradient_to_control_check.md").write_text(
        "# ST-GNN gradient to control check\n\n"
        "ST-GNN-DMPC 在线控制路径为 feature_builder -> trained ST-GNN inference -> OSQP DMPC QP -> UAV dynamics. "
        "逐 UAV 的预测梯度、控制输入、QP 状态和约束状态记录在 `stgnn_closed_loop_trace.csv`。\n",
        encoding="utf-8",
    )
    (debug_dir / "qp_solver_implementation.md").write_text(
        "# QP solver implementation\n\n"
        "ST-GNN-DMPC 使用 OSQP 求解有限时域加速度优化问题。变量为每个 UAV 的 `u[0:H-1] in R^3`，"
        "硬约束包括三轴加速度 box 约束和三轴速度 box 约束。目标函数包含 ST-GNN 预测势场梯度线性项、"
        "位置位移正则、速度惩罚和控制能耗惩罚。QP 失败不会静默替代，失败状态写入 `qp_solver_failure_cases.csv`。\n",
        encoding="utf-8",
    )


def run_simulation_suite(config: SimulationConfig | None = None) -> dict:
    config = (config or SimulationConfig()).resolve()
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(config)
    sample_ids = _load_sample_ids(config)

    results: list[RunResult] = []
    ns3_rows: list[dict] = []
    debug_rows: dict[str, list[dict]] = {"control_trace": [], "qp_rows": [], "allocation_rows": []}
    monitoring_scenarios = [("single_monitoring", 1)] + [
        (_scenario_label(group_size), group_size)
        for group_size in config.concurrent_satellites
        if group_size > 1
    ]
    scenarios = monitoring_scenarios + [("realtime", 1)]
    algorithms = list(ALGORITHM_PROFILES)
    for scenario_index, (scenario_name, group_size) in enumerate(scenarios):
        for algorithm_index, algorithm in enumerate(algorithms):
            for trial in range(config.trials):
                result, trial_ns3_rows, trial_debug_rows = _run_single_trial(
                    algorithm=algorithm,
                    records=records,
                    sample_ids=sample_ids,
                    scenario_name=scenario_name,
                    group_size=group_size,
                    scenario_index=scenario_index,
                    algorithm_index=algorithm_index,
                    trial=trial,
                    config=config,
                )
                results.append(result)
                ns3_rows.extend(trial_ns3_rows)
                for key, rows in trial_debug_rows.items():
                    debug_rows.setdefault(key, []).extend(rows)

    summary: dict[str, dict[str, list[dict]]] = {}
    for scenario_name, _group_size in monitoring_scenarios:
        summary[scenario_name] = {
            metric: _summarize_trials(results, metric, scenario_name).to_dict(orient="records")
            for metric in ("monitoring_performance", "effective_monitoring_ratio", "tracking_error")
        }
    summary["single_network"] = {
        metric: _summarize_trials(results, metric, "single_monitoring").to_dict(orient="records")
        for metric in ("average_connectivity", "connected_graph_ratio", "link_availability_ratio", "k_connectivity_satisfaction_ratio", "link_disruption_rate", "average_node_degree")
    }
    summary["multi3_network"] = {
        metric: _summarize_trials(results, metric, "multi3_monitoring").to_dict(orient="records")
        for metric in ("average_connectivity", "connected_graph_ratio", "link_availability_ratio", "k_connectivity_satisfaction_ratio", "link_disruption_rate", "average_node_degree")
    }
    summary["realtime"] = {
        metric: _summarize_trials(results, metric, "realtime").to_dict(orient="records")
        for metric in ("topology_convergence_latency_ms", "average_single_step_ms", "inference_latency_ms")
    }

    generate_reports(summary, config)
    _write_raw_trial_tables(results, config)
    _write_ns3_dynamic(ns3_rows, config)
    _write_closed_loop_debug(debug_rows, config)
    (config.reports_dir / "simulation_summary_uncalibrated.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
