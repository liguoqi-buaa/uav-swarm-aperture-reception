"""跑独立 baseline（B1 几何 PD / B2 贪心对齐，各有/无 5 步前瞻）。

与 run_scenario_c_retrained.py 同框架，差异：
- 输出目录 reports/scenario_c_independent_{mode}，report_tag=indep_{mode}
- 默认算法为四个独立 baseline（不含 ST-GNN-DMPC，其已有结果可复用）
- 不使用 ST-GNN 权重（这些控制律不含学习组件）

用法：
    python run_independent_baselines.py --mode matched --trials 20 --workers 6
    python run_independent_baselines.py --mode matched --trials 5 --workers 6 \
        --algorithms B1_GEOM_PD__kp4_kd4,B2_GREEDY__kp4_kd4      # 增益扫描
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

DEFAULT_ALGOS = "B1_GEOM_PD,B2_GREEDY,B1_GEOM_PD_H5,B2_GREEDY_H5"

_CTX: dict = {}


def _init_worker(reports_dir, figures_dir, pool, mode, duration_s, export_trials) -> None:
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
        report_tag=f"indep_{mode}",
    ).resolve()
    _CTX["config"] = cfg
    _CTX["export_trials"] = export_trials
    _CTX["records"] = _load_records(cfg)
    _CTX["sample_ids"] = _load_sample_ids(cfg)


def _run_one(job) -> dict:
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
        export_trials=bool(_CTX.get("export_trials", False)),
    )
    row = {
        "scenario": result.scenario,
        "M": result.m,
        "algorithm": result.algorithm,
        "trial_id": result.trial,
        "random_seed": result.random_seed,
        "sample_id": result.sample_id,
        "wall_s": time.perf_counter() - t0,
    }
    row.update({k: v for k, v in result.raw.items()})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="matched", choices=["published", "matched"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--algorithms", type=str, default=DEFAULT_ALGOS)
    ap.add_argument("--suffix", type=str, default="", help="输出目录后缀，供增益扫描区分")
    ap.add_argument("--export-trials", action="store_true", help="导出每 trial 的逐帧轨迹 CSV")
    args = ap.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    jobs = [(a, t) for a in algorithms for t in range(args.trials)]
    tag = f"scenario_c_independent_{args.mode}{args.suffix}"
    reports_dir = ROOT / "reports" / tag
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] mode={args.mode} algos={algorithms} jobs={len(jobs)} workers={args.workers}", flush=True)

    rows: list[dict] = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(reports_dir), str(figures_dir), str(POOL), args.mode, args.duration,
                  bool(args.export_trials)),
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
                print(f"[run] {done}/{len(jobs)} elapsed={el:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["algorithm", "trial_id"]).reset_index(drop=True)
    out_dir = reports_dir / "raw_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scenario_c_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"[run] wrote {csv_path} shape={df.shape}", flush=True)

    metrics = [c for c in df.columns if c not in
               {"scenario", "M", "algorithm", "trial_id", "random_seed", "sample_id", "result_path", "wall_s"}]
    agg = {}
    for algo, sub in df.groupby("algorithm"):
        agg[algo] = {}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            agg[algo][m] = {
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean()) if vals.notna().any() else None,
                "std": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else None,
            }
    (reports_dir / "scenario_c_summary.json").write_text(
        json.dumps({"mode": args.mode, "trials": args.trials, "n_rows": int(len(df)),
                    "by_algorithm": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run] total wall {time.perf_counter()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
