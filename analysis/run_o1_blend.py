"""O1 修复：把 ST-GNN-DMPC 的 connectivity_blend 调回 0 并做 ablation。

背景
----
`src/controllers/stgnn_dmpc_controller.py:311-314`：
    target_traj = (1 - connectivity_blend) * monitoring_target + connectivity_blend * centroid
`config.stgnn_connectivity_blend` 默认 **0.65**，而原 runner 只把 **单星** 版
(`stgnn_connectivity_blend_single_satellite`) 置零，三星场景下仍用 0.65。
于是每个 UAV 的目标被拉向群体质心 65%，编队收缩到 ≈0.35×3000 = 1050 m
（与实测 span 1050 m 完全吻合），覆盖率 C 从 ~0.95 掉到 0.47，直接把 G = A×C 砍半。

本 runner 用 `--blend` 扫描该参数，产出 G/覆盖率/连通度的 trade-off 曲线。

注意：训练标签来自 `composite_potential` 梯度，与该 blend 无关，**无需重训**。

用法：
    python run_o1_blend.py --blend 0.0 --trials 20 --workers 8 --export-trials
    python run_o1_blend.py --blend 0.65 --trials 20 --workers 8
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


def _init_worker(reports_dir, figures_dir, pool, mode, duration_s, model_path, blend, export_trials) -> None:
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
        stgnn_connectivity_blend=blend,               # ← O1 关键：原为 0.65
        stgnn_connectivity_blend_single_satellite=0.0,
        profile_gain_mode=mode,
        report_tag=f"o1b{blend}",
        stgnn_model_path=Path(model_path),
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
    ap.add_argument("--blend", type=float, default=0.0,
                    help="stgnn_connectivity_blend；原始默认 0.65，O1 修复为 0.0")
    ap.add_argument("--model", type=str, default="models/stgnn/best_model_scenario_c.pt")
    ap.add_argument("--algorithms", type=str, default="ST-GNN-DMPC")
    ap.add_argument("--export-trials", action="store_true")
    args = ap.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    jobs = [(a, t) for a in algorithms for t in range(args.trials)]
    tag = f"scenario_c_o1_blend{args.blend}_{args.mode}"
    reports_dir = ROOT / "reports" / tag
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[o1] blend={args.blend} mode={args.mode} model={args.model} "
          f"algos={algorithms} jobs={len(jobs)}", flush=True)

    rows: list[dict] = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(reports_dir), str(figures_dir), str(POOL), args.mode,
                  args.duration, args.model, args.blend, bool(args.export_trials)),
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
                print(f"[o1] {done}/{len(jobs)} elapsed={time.perf_counter()-t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["algorithm", "trial_id"]).reset_index(drop=True)
    out_dir = reports_dir / "raw_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scenario_c_trials.csv", index=False)

    metrics = [c for c in df.columns if c not in
               {"scenario", "M", "algorithm", "trial_id", "random_seed", "sample_id", "result_path", "wall_s"}]
    agg = {}
    for algo, sub in df.groupby("algorithm"):
        agg[algo] = {m: {"mean": float(pd.to_numeric(sub[m], errors="coerce").mean()),
                         "std": float(pd.to_numeric(sub[m], errors="coerce").std(ddof=1))}
                     for m in metrics}
    (reports_dir / "scenario_c_summary.json").write_text(
        json.dumps({"blend": args.blend, "mode": args.mode, "trials": args.trials,
                    "by_algorithm": agg}, indent=2, ensure_ascii=False), encoding="utf-8")

    k = ["aperture_gain", "aperture_alignment", "aperture_coverage", "aperture_span_m",
         "aperture_misalignment_deg", "average_connectivity", "link_availability_ratio",
         "average_single_step_ms"]
    print(f"[o1] blend={args.blend}  " + "  ".join(f"{x}={agg['ST-GNN-DMPC'][x]['mean']:.4f}" for x in k if x in agg["ST-GNN-DMPC"]))
    print(f"[o1] total wall {time.perf_counter()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
