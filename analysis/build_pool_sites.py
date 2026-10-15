"""按站点生成共视窗口池（第二/第三站点鲁棒性检验）。

由 `build_covisible_pool2.py` 改写：去掉硬编码的旧工作区路径，改为命令行参数化。
逻辑与算法与 2 版完全一致（按窗口内累计仰角选 3 星 + keyhole 排除 + 10 s 细筛），
以保证跨站点可比。

用法
----
    python build_pool_sites.py --sites "mid30_west:30.0:-100.0" "mid45_europe:45.0:8.0"
    python build_pool_sites.py --sites "eq_pacific:0.0:-174.624885" --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "uav_swarm_scenario_C"
OUT_DIR = HERE / "pools"
sys.path.insert(0, str(ROOT))

from tle_swarm_sim.sgp4_utils import geodetic_to_ecef, satellite_ecef_at  # noqa: E402
from tle_swarm_sim.tle import load_tle_records  # noqa: E402

R_E = 6378137.0
WINDOW_S = 600.0
MIN_ELEV_DEG = 10.0
MIN_SUBPOINT_M = 300_000.0
M_SAT = 3
COARSE_DT = 60.0
START_STEP_MIN = 10.0
TARGET_WINDOWS = 40
MIN_VALID_FRACTION = 0.50


def build(site_lat: float, site_lon: float) -> tuple[dict, dict]:
    archive = ROOT / "data" / "processed" / "starlink_archive.tle"
    recs = load_tle_records(archive)
    print(f"  records={len(recs)}  (archive={archive.name})", flush=True)

    la, lo = np.radians(site_lat), np.radians(site_lon)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    site_ecef = geodetic_to_ecef(site_lat, site_lon, 0.0)

    epochs = [r.epoch_datetime for r in recs]
    t0 = min(epochs).replace(tzinfo=None)
    t1 = max(epochs).replace(tzinfo=None)
    span_start = t0 - timedelta(minutes=30)
    span_end = t1 + timedelta(minutes=60)
    coarse_times = []
    t = span_start
    while t <= span_end:
        coarse_times.append(t)
        t += timedelta(seconds=COARSE_DT)

    n_rec, n_t = len(recs), len(coarse_times)
    elev = np.full((n_rec, n_t), -90.0, dtype=np.float32)
    ground_km = np.full((n_rec, n_t), 0.0, dtype=np.float32)
    for i, rec in enumerate(recs):
        for j, ts in enumerate(coarse_times):
            r = np.asarray(satellite_ecef_at(rec, ts), dtype=float)
            d = r - site_ecef
            nd = np.linalg.norm(d)
            if nd < 1.0:
                continue
            elev[i, j] = np.degrees(np.arcsin(np.clip(float(d @ up) / nd, -1.0, 1.0)))
            cosg = float(r @ up) / max(np.linalg.norm(r), 1.0)
            ground_km[i, j] = R_E * np.arccos(np.clip(cosg, -1.0, 1.0)) / 1e3

    gain = np.clip(elev - MIN_ELEV_DEG, 0.0, None).astype(np.float64)
    steps_per_window = int(WINDOW_S / COARSE_DT) + 1

    starts = []
    t = span_start + timedelta(minutes=5)
    while t + timedelta(seconds=WINDOW_S) <= span_end:
        starts.append(t)
        t += timedelta(minutes=START_STEP_MIN)

    windows = []
    pre = []
    for st in starts:
        j0 = int(round((st - span_start).total_seconds() / COARSE_DT))
        j1 = j0 + steps_per_window
        if j1 > n_t:
            continue
        off_axis = ground_km[:, j0:j1].min(axis=1) * 1e3 >= MIN_SUBPOINT_M
        score = np.where(off_axis, gain[:, j0:j1].sum(axis=1), -1.0)
        order = np.argsort(-score)
        cats, ok = [], True
        for idx in order[:M_SAT]:
            if score[idx] <= 0.0:
                ok = False
                break
            cats.append(str(recs[int(idx)].catalog_number))
        if not ok or len(cats) < 2:
            continue
        fine = [st + timedelta(seconds=i * 10.0) for i in range(int(WINDOW_S / 10.0) + 1)]
        el_f = np.zeros((len(cats), len(fine)))
        dmin = np.inf
        for m, cat in enumerate(cats):
            rec = next(r for r in recs if str(r.catalog_number) == cat)
            for k, ts in enumerate(fine):
                r = np.asarray(satellite_ecef_at(rec, ts), dtype=float)
                d = r - site_ecef
                nd = np.linalg.norm(d)
                el_f[m, k] = np.degrees(np.arcsin(np.clip(float(d @ up) / nd, -1.0, 1.0)))
                cosg = float(r @ up) / max(np.linalg.norm(r), 1.0)
                dmin = min(dmin, R_E * np.arccos(np.clip(cosg, -1.0, 1.0)))
        vf = float((np.clip(el_f - MIN_ELEV_DEG, 0, None).sum(axis=0) > 1e-9).mean())
        pre.append((vf, dmin / 1e3))
        windows.append({
            "start_utc": st.isoformat(),
            "catalogs": cats,
            "valid_fraction_10deg": round(vf, 4),
            "min_subpoint_km": round(dmin / 1e3, 1),
            "elev_start_deg": [round(float(el_f[m, 0]), 2) for m in range(len(cats))],
            "mean_gain_deg": round(float(np.clip(el_f - MIN_ELEV_DEG, 0, None).mean()), 2),
        })

    arr = np.asarray(pre) if pre else np.zeros((0, 2))
    n_pass = sum(1 for w in windows
                 if w["min_subpoint_km"] * 1e3 >= MIN_SUBPOINT_M
                 and w["valid_fraction_10deg"] >= MIN_VALID_FRACTION)
    diag = {
        "n_records": n_rec,
        "span_start": span_start.isoformat(),
        "span_end": span_end.isoformat(),
        "n_coarse_points": n_t,
        "n_candidate_starts": len(starts),
        "n_windows_built": len(windows),
        "n_windows_pass_geometry": int(n_pass),
        "vf_min": float(arr[:, 0].min()) if len(arr) else None,
        "vf_median": float(np.median(arr[:, 0])) if len(arr) else None,
        "dmin_min_km": float(arr[:, 1].min()) if len(arr) else None,
        "dmin_median_km": float(np.median(arr[:, 1])) if len(arr) else None,
    }

    windows = [w for w in windows
               if w["min_subpoint_km"] * 1e3 >= MIN_SUBPOINT_M
               and w["valid_fraction_10deg"] >= MIN_VALID_FRACTION]
    windows.sort(key=lambda w: (-w["valid_fraction_10deg"], -w["mean_gain_deg"]))
    n_usable = len(windows)
    windows = windows[:TARGET_WINDOWS]

    pool = {
        "site_lat_deg": site_lat,
        "site_lon_deg": site_lon,
        "min_elevation_deg": MIN_ELEV_DEG,
        "window_s": WINDOW_S,
        "m_sat": M_SAT,
        "windows": windows,
    }
    diag["n_windows_usable"] = n_usable
    diag["n_windows_kept"] = len(windows)
    return pool, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", required=True,
                    help='形如 "name:lat:lon"，可给多个')
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in args.sites:
        name, slat, slon = spec.split(":")
        lat, lon = float(slat), float(slon)
        print(f"[pool] {name}: lat={lat} lon={lon}", flush=True)
        t0 = time.perf_counter()
        pool, diag = build(lat, lon)
        el = time.perf_counter() - t0
        diag["build_seconds"] = round(el, 1)
        tag = f"_{args.tag}" if args.tag else ""
        out = OUT_DIR / f"covisible_pool_{name}{tag}.json"
        if not args.check_only:
            out.write_text(json.dumps(pool, indent=2), encoding="utf-8")
        summary[name] = {"pool_path": str(out), **diag}
        print(f"  -> {len(pool['windows'])} windows kept (usable={diag['n_windows_usable']}, "
              f"built={diag['n_windows_built']}) in {el:.0f}s", flush=True)

    (OUT_DIR / "pool_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== pool build summary ===")
    for k, v in summary.items():
        print(f"{k:16s} built={v['n_windows_built']:4d} usable={v['n_windows_usable']:4d} "
              f"kept={v['n_windows_kept']:3d}  vf_med={v['vf_median']}  "
              f"dmin_med={v['dmin_median_km']} km  {v['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
