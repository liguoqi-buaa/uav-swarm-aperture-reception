"""Scenario C full run: 5 algorithms x N trials, parallel over trials.

用法（在 scenario_C 工作副本目录下，用 venv 解释器）：
    python analysis/run_scenario_c.py --mode published --trials 20 --workers 10
    python analysis/run_scenario_c.py --mode matched   --trials 20 --workers 10

产出：
    <ROOT>\reports\scenario_c_<mode>\trials\...            逐 trial 产物
    <ROOT>\reports\scenario_c_<mode>\raw_trials\scenario_c_trials.csv
    <ROOT>\reports\scenario_c_<mode>\scenario_c_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "uav_swarm_scenario_C"
sys.path.insert(0, str(ROOT))

POOL = Path(__file__).resolve().parent / "covisible_pool.json"

_CTX: dict = {}


def _init_worker(reports_dir: str, figures_dir: str, pool: str, mode: str, duration_s: float) -> None:
    import torch

    torch.set_num_threads(1)
    from tle_swarm_sim.simulation.config import SimulationConfig
    from tle_swarm_sim.simulation.experiment import _load_records, _load_sample_ids

    cfg = SimulationConfig(
        repo_root=ROOT,
        reports_dir=Path(reports_dir),
        figures_dir=Path(figures_dir),
        trials=20,
        concurrent_satellites=(3,),
        simulation_duration_s=duration_s,
        control_step_s=1.0,
        scenario_mode="aperture",
        site_lat_deg=0.0,
        site_lon_deg=-174.624885,
        aperture_pool_path=Path(pool),
        aperture_baseline_m=3000.0,
        aperture_sigma_m=500.0,
        aperture_min_elevation_deg=10.0,
        monitoring_kernel_radius_m=1000.0,
        deployment_mode="monitor_area_near",
        monitor_area_initial_spread_m=1500.0,
        slot_smoothing_alpha=1.0,
        max_communication_range_m=2500.0,
        task_window_mode="extended_pass_10deg",
        stgnn_connectivity_blend_single_satellite=0.0,
        profile_gain_mode=mode,
        report_tag=mode,
    ).resolve()
    _CTX["config"] = cfg
    _CTX["records"] = _load_records(cfg)
    _CTX["sample_ids"] = _load_sample_ids(cfg)
    _CTX["mode"] = mode


def _run_one(job: tuple[str, int]) -> dict:
    from tle_swarm_sim.simulation.experiment import _run_single_trial

    algorithm, trial = job
    cfg = _CTX["config"]
    t0 = time.perf_counter()
    result, _ns3, _dbg = _run_single_trial(
        algorithm=algorithm,
        records=_CTX["records"],
        sample_ids=_CTX["sample_ids"],
        scenario_name="aperture_M3",
        group_size=3,
        scenario_index=0,
        algorithm_index=0,
        trial=trial,
        config=cfg,
        export_trials=True,
    )
    row = {
        "scenario": result.scenario,
        "M": result.m,
        "algorithm": result.algorithm,
        "trial_id": result.trial,
        "random_seed": result.random_seed,
        "sample_id": result.sample_id,
        "result_path": result.result_path,
        "wall_s": time.perf_counter() - t0,
    }
    row.update({k: v for k, v in result.raw.items()})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="published", choices=["published", "matched"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--duration", type=float, default=600.0)
    args = ap.parse_args()

    reports_dir = ROOT / "reports" / f"scenario_c_{args.mode}"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from tle_swarm_sim.simulation.controllers import ALGORITHM_PROFILES

    algorithms = list(ALGORITHM_PROFILES)
    jobs = [(a, t) for a in algorithms for t in range(args.trials)]
    print(f"[run] mode={args.mode} trials={args.trials} algos={len(algorithms)} "
          f"jobs={len(jobs)} workers={args.workers} duration={args.duration}s", flush=True)

    rows: list[dict] = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(reports_dir), str(figures_dir), str(POOL), args.mode, args.duration),
    ) as ex:
        futures = {ex.submit(_run_one, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            done += 1
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {job}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if done % 5 == 0 or done == len(jobs):
                el = time.perf_counter() - t_start
                print(f"[run] {done}/{len(jobs)} elapsed={el:.0f}s eta={el/done*(len(jobs)-done):.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["algorithm", "trial_id"]).reset_index(drop=True)
    out_dir = reports_dir / "raw_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scenario_c_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"[run] wrote {csv_path}  shape={df.shape}", flush=True)

    metrics = [c for c in df.columns if c not in
               {"scenario", "M", "algorithm", "trial_id", "random_seed", "sample_id", "result_path"}]
    agg = {}
    for algo, sub in df.groupby("algorithm"):
        agg[algo] = {}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            agg[algo][m] = {
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean()) if vals.notna().any() else None,
                "std": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else None,
                "min": float(vals.min()) if vals.notna().any() else None,
                "max": float(vals.max()) if vals.notna().any() else None,
            }
    (reports_dir / "scenario_c_summary.json").write_text(
        json.dumps({"mode": args.mode, "trials": args.trials, "duration_s": args.duration,
                    "n_rows": int(len(df)), "by_algorithm": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run] total wall {time.perf_counter()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
