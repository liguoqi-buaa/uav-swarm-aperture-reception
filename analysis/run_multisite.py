"""多站点闭环 runner（第二/第三站点鲁棒性检验）。

与 `run_scenario_c.py` 一致，但：
  * 站点（lat/lon）与窗口池由命令行给出；
  * 同一轮里同时跑「同律参数族 P0–P3」「预测族 ST-GNN-DMPC」「独立几何族 B1/B2」
    以及两个滚动时域预览变体，保证 9 个控制器共用同一批窗口与随机种子；
  * 输出到 `reports/multisite_<site>/`，并在 CSV 里带上站点列。

注意：`scenario_c.aperture_geometry()` 读的是 **config.site_lat/lon**（不是池子里
的），所以两者必须给成一致的值，否则几何与窗口池会对不上。

用法
----
    python run_multisite.py --site-name mid30_west --lat 30.0 --lon -100.0 \
        --pool pools/covisible_pool_mid30_west.json --trials 20 --workers 8
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "uav_swarm_scenario_C"
sys.path.insert(0, str(ROOT))

DEFAULT_ALGOS = (
    "OLSR-FANET,APF-TC,GNN-TC,DMPC-C,ST-GNN-DMPC,"
    "B1_GEOM_PD,B2_GREEDY,B1_GEOM_PD_PREV__w2,B2_GREEDY_PREV__w2"
)

_CTX: dict = {}


def _init_worker(reports_dir: str, figures_dir: str, pool: str, mode: str,
                 duration_s: float, site_lat: float, site_lon: float, model: str) -> None:
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
        site_lat_deg=site_lat,
        site_lon_deg=site_lon,
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
        stgnn_model_path=Path(model),
        profile_gain_mode=mode,
        report_tag=mode,
    ).resolve()
    _CTX["config"] = cfg
    _CTX["records"] = _load_records(cfg)
    _CTX["sample_ids"] = _load_sample_ids(cfg)


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
    ap.add_argument("--site-name", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--mode", default="matched", choices=["published", "matched"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--model", default="models/stgnn/best_model.pt")
    ap.add_argument("--algorithms", default=DEFAULT_ALGOS)
    args = ap.parse_args()

    reports_dir = ROOT / "reports" / f"multisite_{args.site_name}"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    jobs = [(a, t) for a in algorithms for t in range(args.trials)]
    print(f"[run] site={args.site_name} ({args.lat},{args.lon}) mode={args.mode} "
          f"trials={args.trials} algos={len(algorithms)} jobs={len(jobs)} "
          f"workers={args.workers} model={args.model}", flush=True)

    rows: list[dict] = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(reports_dir), str(figures_dir), str(Path(args.pool).resolve()),
                  args.mode, args.duration, args.lat, args.lon, args.model),
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
            if done % 20 == 0 or done == len(jobs):
                el = time.perf_counter() - t_start
                print(f"[run] {done}/{len(jobs)} elapsed={el:.0f}s "
                      f"eta={el/done*(len(jobs)-done):.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["algorithm", "trial_id"]).reset_index(drop=True)
    df.insert(0, "site", args.site_name)
    df.insert(1, "site_lat_deg", args.lat)
    df.insert(2, "site_lon_deg", args.lon)
    out_dir = reports_dir / "raw_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scenario_c_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"[run] wrote {csv_path}  shape={df.shape}", flush=True)

    metrics = [c for c in df.columns if c not in
               {"site", "site_lat_deg", "site_lon_deg", "scenario", "M", "algorithm",
                "trial_id", "random_seed", "sample_id", "result_path"}]
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
        json.dumps({"site_name": args.site_name, "site_lat_deg": args.lat,
                    "site_lon_deg": args.lon, "pool": str(Path(args.pool).resolve()),
                    "mode": args.mode, "trials": args.trials, "duration_s": args.duration,
                    "model": args.model, "n_rows": int(len(df)),
                    "by_algorithm": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run] total wall {time.perf_counter()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
