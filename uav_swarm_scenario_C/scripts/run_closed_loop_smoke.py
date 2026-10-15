from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tle_swarm_sim.simulation.config import SimulationConfig
from tle_swarm_sim.simulation.experiment import _load_records, _load_sample_ids, _run_single_trial


def _mean(rows: list[dict], metric: str, scenario: str, algorithm: str) -> float:
    vals = [r[metric] for r in rows if r["scenario"] == scenario and r["algorithm"] == algorithm]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    config = SimulationConfig(trials=3).resolve()
    smoke_dir = config.reports_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(config)
    sample_ids = _load_sample_ids(config)
    algorithms = ["GNN-TC", "DMPC-C", "ST-GNN-DMPC"]
    scenarios = [("single_monitoring", 1), ("multi3_monitoring", 3)]
    rows = []
    debug = {"control_trace": [], "qp_rows": [], "allocation_rows": []}
    for scenario_index, (scenario_name, group_size) in enumerate(scenarios):
        for algorithm_index, algorithm in enumerate(algorithms):
            for trial in range(3):
                result, _ns3, trial_debug = _run_single_trial(
                    algorithm=algorithm,
                    records=records,
                    sample_ids=sample_ids,
                    scenario_name=scenario_name,
                    group_size=group_size,
                    scenario_index=scenario_index,
                    algorithm_index=algorithm_index,
                    trial=trial,
                    config=config,
                    export_trials=True,
                )
                rows.append(
                    {
                        "scenario": result.scenario,
                        "M": result.m,
                        "algorithm": result.algorithm,
                        "trial_id": result.trial,
                        "random_seed": result.random_seed,
                        **result.raw,
                    }
                )
                for key, value in trial_debug.items():
                    debug.setdefault(key, []).extend(value)
    df = pd.DataFrame(rows)
    df.to_csv(smoke_dir / "stgnn_dmpc_closed_loop_smoke_trials.csv", index=False)
    if debug["control_trace"]:
        pd.DataFrame(debug["control_trace"]).to_csv(config.reports_dir / "debug" / "stgnn_closed_loop_trace.csv", index=False)
    if debug["qp_rows"]:
        pd.DataFrame(debug["qp_rows"]).to_csv(config.reports_dir / "debug" / "qp_solver_status_summary.csv", index=False)
    if debug["allocation_rows"]:
        pd.DataFrame(debug["allocation_rows"]).to_csv(config.reports_dir / "debug" / "task_allocation_runtime_trace.csv", index=False)

    st = df[df["algorithm"] == "ST-GNN-DMPC"]
    qp_solved_ratio = float(st["qp_solved_ratio"].mean())
    fallback_rate = float(st["fallback_rate"].mean())
    clip_ratio = float(st["acceleration_clip_ratio"].mean())
    stgnn_loaded = bool((st["stgnn_loaded"] == 1.0).all())
    checks = {
        "stgnn_loaded": stgnn_loaded,
        "qp_solved_ratio_ge_95pct": qp_solved_ratio >= 0.95,
        "fallback_rate_le_5pct": fallback_rate <= 0.05,
        "acceleration_clip_ratio_lt_30pct": clip_ratio < 0.30,
        "M1_beats_DMPC_J": _mean(rows, "monitoring_performance", "single_monitoring", "ST-GNN-DMPC")
        > _mean(rows, "monitoring_performance", "single_monitoring", "DMPC-C"),
        "M3_beats_DMPC_J": _mean(rows, "monitoring_performance", "multi3_monitoring", "ST-GNN-DMPC")
        > _mean(rows, "monitoring_performance", "multi3_monitoring", "DMPC-C"),
        "M1_beats_DMPC_tracking_error": _mean(rows, "tracking_error", "single_monitoring", "ST-GNN-DMPC")
        < _mean(rows, "tracking_error", "single_monitoring", "DMPC-C"),
        "M3_beats_DMPC_tracking_error": _mean(rows, "tracking_error", "multi3_monitoring", "ST-GNN-DMPC")
        < _mean(rows, "tracking_error", "multi3_monitoring", "DMPC-C"),
    }
    passed = all(checks.values())
    means = (
        df.groupby(["scenario", "algorithm"])
        .agg(
            monitoring_performance=("monitoring_performance", "mean"),
            effective_monitoring_ratio=("effective_monitoring_ratio", "mean"),
            tracking_error=("tracking_error", "mean"),
            qp_solved_ratio=("qp_solved_ratio", "mean"),
            fallback_rate=("fallback_rate", "mean"),
            acceleration_clip_ratio=("acceleration_clip_ratio", "mean"),
        )
        .reset_index()
    )
    report = [
        "# ST-GNN-DMPC closed-loop smoke report",
        "",
        f"overall_status: {'pass' if passed else 'fail'}",
        f"stgnn_loaded: {stgnn_loaded}",
        f"qp_solved_ratio: {qp_solved_ratio:.4f}",
        f"fallback_rate: {fallback_rate:.4f}",
        f"acceleration_clip_ratio: {clip_ratio:.4f}",
        "",
        "## Checks",
        "",
        pd.DataFrame([{"check": k, "pass": v} for k, v in checks.items()]).to_markdown(index=False),
        "",
        "## Mean metrics",
        "",
        means.to_markdown(index=False),
    ]
    (smoke_dir / "stgnn_dmpc_closed_loop_smoke_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "pass" if passed else "fail", "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
