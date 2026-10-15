"""跨站点鲁棒性分析：赤道站点（参照） + 30°N + 45°N。

检验四条论文结论是否只对单一站点成立：
  R1 长度混杂：等长度 A 相对实测 A 的回落幅度
  R2 排序：B2 是否在三个站点上都胜过 ST-GNN-DMPC（A_m / A_L0 / A_dir / G）
  R3 前瞻无益：B2^p 相对 B2 的 G 变化是否在三个站点上都为负
  R4 连通度不区分：λ2 的量级与跨站点漂移
  R5 参考次优：v_ref 与瞬时最优 v_demand 的夹角 >30° 的时间占比

输出：`多站点鲁棒性检验_报告.md` 与 `analysis/multisite_summary.csv`。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "uav_swarm_scenario_C"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from normalize_alignment import per_trial_metrics  # noqa: E402

# 与论文一致：权重参考仰角用 10°（= config.aperture_min_elevation_deg），
# 而不是 normalize_alignment.py 里遗留的默认值 15°。
# 校验：MIN_ELEV=10 时赤道站点复算得 P3 A_L0=0.2981 / ST=0.5238，
# 与稿件 tab:benchmark 的 0.298 / 0.524 完全一致。
import normalize_alignment as _NA  # noqa: E402
_NA.MIN_ELEV = 10.0
MIN_ELEV_USED = 10.0

CORE = ["OLSR-FANET", "APF-TC", "GNN-TC", "DMPC-C", "ST-GNN-DMPC", "B1_GEOM_PD", "B2_GREEDY"]
PREV = ["B1_GEOM_PD_PREV__w2", "B2_GREEDY_PREV__w2"]
LABEL = {"OLSR-FANET": "P0", "APF-TC": "P1", "GNN-TC": "P2", "DMPC-C": "P3",
         "ST-GNN-DMPC": "ST-GNN-DMPC", "B1_GEOM_PD": "B1", "B2_GREEDY": "B2",
         "B1_GEOM_PD_PREV__w2": "B1p", "B2_GREEDY_PREV__w2": "B2p"}

SITES = [
    dict(name="eq_pacific", pretty="赤道太平洋 (0.0N, 174.62W)", lat=0.0, lon=-174.624885,
         pool=str(HERE / "covisible_pool.json"),
         dirs=["scenario_c_matched", "scenario_c_independent_matched_full",
               "scenario_c_independent_matched_prev"]),
    dict(name="mid30_west", pretty="中纬 30N (30.0N, 100.0W)", lat=30.0, lon=-100.0,
         pool=str(HERE / "pools" / "covisible_pool_mid30_west.json"),
         dirs=["multisite_mid30_west"]),
    dict(name="mid45_europe", pretty="中纬 45N (45.0N, 8.0E)", lat=45.0, lon=8.0,
         pool=str(HERE / "pools" / "covisible_pool_mid45_europe.json"),
         dirs=["multisite_mid45_europe"]),
]

KEEP = ["aperture_gain", "aperture_alignment", "aperture_coverage",
        "aperture_span_m", "aperture_misalignment_deg", "average_connectivity",
        "link_availability_ratio", "average_single_step_ms", "task_valid_fraction",
        "elev_mean_deg", "subpoint_dist_mean_km", "tracking_error",
        "aperture_wavefront_rms_m"]


# ---------------------------------------------------------------------------
# 1) 汇总各站点的 trial 级指标
# ---------------------------------------------------------------------------
def collect_site(site: dict) -> pd.DataFrame:
    frames = []
    for d in site["dirs"]:
        p = ROOT / "reports" / d / "raw_trials" / "scenario_c_trials.csv"
        if not p.exists():
            print(f"  [WARN] 缺 {p}")
            continue
        df = pd.read_csv(p)
        cols = ["algorithm", "trial_id"] + [c for c in KEEP if c in df.columns]
        frames.append(df[cols])
    df = pd.concat(frames, ignore_index=True)
    # 去重：同一 (algo, trial) 出现在多个目录时，保留第一个目录的版本
    df["_ord"] = df["algorithm"].map({a: i for i, a in enumerate(CORE + PREV)}).fillna(99)
    df = df.sort_values("_ord").drop_duplicates(["algorithm", "trial_id"], keep="first")
    df = df.drop(columns=["_ord"]).reset_index(drop=True)
    df["site"] = site["name"]
    return df


def add_normalized(site: dict, df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (站点, 算法) 的 trial 目录复算 A@3000 / Adir@3000。"""
    recs = []
    for algo in CORE + PREV:
        for d in site["dirs"]:
            algo_dir = ROOT / "reports" / d / "trials" / "M3" / algo
            if not algo_dir.exists():
                continue
            for trial_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
                try:
                    r = per_trial_metrics(trial_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [WARN] {trial_dir.name}: {type(exc).__name__}: {exc}")
                    continue
                r["algorithm"] = algo
                r["trial_id"] = int(trial_dir.name.rsplit("_", 1)[-1])
                recs.append(r)
            break  # 只用第一个含该算法的目录
    nd = pd.DataFrame(recs)
    nd = nd.rename(columns={"A@3000": "A_L0", "Adir@3000": "A_dir"})
    keep = ["algorithm", "trial_id", "A_L0", "A_dir", "A_raw_all", "valid_fraction"]
    nd = nd[[c for c in keep if c in nd.columns]]
    return df.merge(nd, on=["algorithm", "trial_id"], how="left")


# ---------------------------------------------------------------------------
# 2) 参考次优（纯几何，可从窗口池重建）
# ---------------------------------------------------------------------------
def reference_stats(site: dict, trials: int = 20) -> dict:
    from datetime import datetime

    from tle_swarm_sim.simulation.config import SimulationConfig
    from tle_swarm_sim.simulation.experiment import _load_records
    from tle_swarm_sim.simulation.scenario_c import (
        build_aperture_task, group_records_for_window, select_aperture_window)

    cfg = SimulationConfig(
        repo_root=ROOT, reports_dir=ROOT / "reports", figures_dir=ROOT / "reports" / "figures",
        trials=20, concurrent_satellites=(3,), simulation_duration_s=600.0, control_step_s=1.0,
        scenario_mode="aperture", site_lat_deg=site["lat"], site_lon_deg=site["lon"],
        aperture_pool_path=Path(site["pool"]), aperture_baseline_m=3000.0,
        aperture_sigma_m=500.0, aperture_min_elevation_deg=10.0,
        monitoring_kernel_radius_m=1000.0, deployment_mode="monitor_area_near",
        monitor_area_initial_spread_m=1500.0, slot_smoothing_alpha=1.0,
        max_communication_range_m=2500.0, task_window_mode="extended_pass_10deg",
        stgnn_connectivity_blend_single_satellite=0.0, profile_gain_mode="matched",
    ).resolve()
    records = _load_records(cfg)

    angles = []
    windows = set()
    for trial in range(trials):
        w = select_aperture_window(records, cfg, trial)
        windows.add(w["start_utc"])
        grp = group_records_for_window(records, w)
        _task, geo = build_aperture_task(grp, datetime.fromisoformat(w["start_utc"]), cfg)
        vd = np.asarray(geo["v_demand"], dtype=float)   # (T,2)
        vr = np.asarray(geo["v_star"], dtype=float)     # (T,2)
        valid = np.asarray(geo["valid"], dtype=bool)
        cos = np.abs(np.sum(vd * vr, axis=1)) / np.maximum(
            np.linalg.norm(vd, axis=1) * np.linalg.norm(vr, axis=1), 1e-12)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))   # 无向夹角 0-90
        angles.append(ang[valid])
    a = np.concatenate(angles) if angles else np.array([np.nan])
    return {
        "site": site["name"],
        "n_windows_used": len(windows),
        "angle_mean_deg": float(np.nanmean(a)),
        "angle_median_deg": float(np.nanmedian(a)),
        "angle_p90_deg": float(np.nanpercentile(a, 90)),
        "angle_max_deg": float(np.nanmax(a)),
        "frac_gt30": float((a > 30).mean()),
        "frac_gt45": float((a > 45).mean()),
        "n_samples": int(a.size),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-geometry", action="store_true")
    args = ap.parse_args()

    per_site: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for site in SITES:
        print(f"[analyze] {site['name']} ...", flush=True)
        df = collect_site(site)
        df = add_normalized(site, df)
        per_site[site["name"]] = df
        pool = json.loads(Path(site["pool"]).read_text(encoding="utf-8"))
        meta[site["name"]] = {
            "pretty": site["pretty"], "lat": site["lat"], "lon": site["lon"],
            "pool_windows": len(pool["windows"]),
            "n_trials": int(df["trial_id"].nunique()),
        }
        print(f"    {df.shape}, trials={df['trial_id'].nunique()}", flush=True)

    # 汇总表
    rows = []
    for sname, df in per_site.items():
        for algo, sub in df.groupby("algorithm"):
            r = {"site": sname, "algorithm": algo, "label": LABEL.get(algo, algo), "n": len(sub)}
            for c in ["aperture_gain", "aperture_alignment", "A_L0", "A_dir",
                      "aperture_coverage", "aperture_span_m", "aperture_misalignment_deg",
                      "average_connectivity", "link_availability_ratio",
                      "average_single_step_ms", "task_valid_fraction", "elev_mean_deg",
                      "subpoint_dist_mean_km", "subpoint_dist_min_km"]:
                if c in sub.columns:
                    v = pd.to_numeric(sub[c], errors="coerce")
                    r[c + "_mean"] = float(v.mean())
                    r[c + "_std"] = float(v.std(ddof=1))
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(HERE / "multisite_summary.csv", index=False)

    ref = None if args.skip_geometry else pd.DataFrame([reference_stats(s) for s in SITES])
    if ref is not None:
        ref.to_csv(HERE / "multisite_reference_stats.csv", index=False)

    # ---------------- Markdown ----------------
    md: list[str] = []
    md.append("# 多站点鲁棒性检验（赤道参照 + 30°N + 45°N）\n")
    md.append("生成脚本：`analysis/analyze_multisite.py`；逐 trial 数据来自 "
              "`uav_swarm_scenario_C/reports/`；等长度 A 由 `normalize_alignment.py` 复算，"
              f"权重参考仰角取 {MIN_ELEV_USED:.0f}°（= `config.aperture_min_elevation_deg`，"
              "与稿件 `tab:benchmark` 同口径）。\n")

    md.append("## 1. 站点与样本\n")
    md.append("| 站点 | 纬度 | 经度 | 池内窗口 | 实际 trial | TLE 归档 |")
    md.append("|:--|--:|--:|--:|--:|:--|")
    for s in SITES:
        m = meta[s["name"]]
        md.append(f"| {m['pretty']} | {m['lat']:.1f}° | {m['lon']:.2f}° | "
                  f"{m['pool_windows']} | {m['n_trials']} | 同一份 1,329 条主壳层归档 |")
    md.append("")
    md.append("> 三个站点使用**同一份 TLE 归档**，只改变观测站经纬度并重算窗口池；"
              "每个站点 20 个 trial（配对协议：同一 trial 序号对所有控制器取同一个窗口）。\n")
    md.append("> 说明：`rank_windows` 按窗口内最大视线转角降序排列，"
              "`trial % len(ranked)` 取窗，因此 20 个 trial 实际只覆盖排名前 20 的窗口。\n")

    # 主表
    md.append("## 2. 跨站点主指标（均值 ± 标准差，n=20）\n")
    for key, title in [("aperture_gain", "孔径增益 $G$"),
                       ("aperture_alignment", "实测对齐质量 $A_m$"),
                       ("A_L0", "等长度对齐质量 $A_{L_0}$ (3000 m)"),
                       ("A_dir", "纯朝向对齐质量 $A_{dir}$ (3000 m)"),
                       ("aperture_span_m", "阵列跨度 (m)"),
                       ("average_connectivity", r"代数连通度 $\lambda_2$"),
                       ("average_single_step_ms", "单步耗时 (ms)")]:
        md.append(f"### {title}\n")
        md.append("| 控制器 | " + " | ".join(meta[s["name"]]["pretty"] for s in SITES) + " |")
        md.append("|:--|" + "--:|" * len(SITES))
        for algo in CORE + PREV:
            cells = []
            for s in SITES:
                sub = summ[(summ.site == s["name"]) & (summ.algorithm == algo)]
                if sub.empty or f"{key}_mean" not in sub:
                    cells.append("—")
                    continue
                m = sub[f"{key}_mean"].iloc[0]
                sd = sub.get(f"{key}_std", pd.Series([np.nan])).iloc[0]
                if not np.isfinite(m):
                    cells.append("—")
                elif key == "average_single_step_ms":
                    cells.append(f"{m:.2f} ± {sd:.2f}")
                elif key == "aperture_span_m":
                    cells.append(f"{m:.0f} ± {sd:.0f}")
                else:
                    cells.append(f"{m:.3f} ± {sd:.3f}")
            md.append(f"| {LABEL.get(algo, algo)} | " + " | ".join(cells) + " |")
        md.append("")

    # R1
    md.append("## 3. R1 长度混杂是否跨站点存在\n")
    md.append("实测 $A_m$ 与等长度 $A_{L_0}$ 之差即「长度效应」所占份额。"
              "两个非学习控制器：预测型与最好的同律档位 $P_3$。\n")
    md.append("| 站点 | $A_m$ (ST) | $A_{L_0}$ (ST) | 长度贡献占比 | $A_m$ (P3) | $A_{L_0}$ (P3) | 长度贡献占比 |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|")
    for s in SITES:
        def g(algo, col):
            sub = summ[(summ.site == s["name"]) & (summ.algorithm == algo)]
            return float(sub[col + "_mean"].iloc[0]) if not sub.empty else np.nan
        st_m, st_l = g("ST-GNN-DMPC", "aperture_alignment"), g("ST-GNN-DMPC", "A_L0")
        p3_m, p3_l = g("DMPC-C", "aperture_alignment"), g("DMPC-C", "A_L0")
        f = lambda a, b: 100 * (a - b) / a if np.isfinite(a) and a > 0 else np.nan
        md.append(f"| {meta[s['name']]['pretty']} | {st_m:.3f} | {st_l:.3f} | "
                  f"{f(st_m, st_l):.1f}% | {p3_m:.3f} | {p3_l:.3f} | {f(p3_m, p3_l):.1f}% |")
    md.append("")

    # R2
    md.append("## 4. R2 控制器排序是否跨站点一致（$ST$ vs $B_2$）\n")
    md.append("| 站点 | 指标 | ST-GNN-DMPC | B2 | 相对差 | 配对 $t$ 检验 $p$ | 排序 |")
    md.append("|:--|:--|--:|--:|--:|--:|:--|")
    order_flips = []
    for s in SITES:
        df = per_site[s["name"]]
        for col, title in [("aperture_gain", "G"), ("aperture_alignment", "A_m"),
                           ("A_L0", "A_L0"), ("A_dir", "A_dir")]:
            a = df[df.algorithm == "ST-GNN-DMPC"].sort_values("trial_id")[col].to_numpy(float)
            b = df[df.algorithm == "B2_GREEDY"].sort_values("trial_id")[col].to_numpy(float)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            p = np.nan
            try:
                from scipy import stats as _st
                p = float(_st.ttest_rel(a, b).pvalue)
            except Exception:
                pass
            rel = 100 * (np.nanmean(a) - np.nanmean(b)) / np.nanmean(b)
            who = "ST 高" if rel > 0 else "B2 高"
            order_flips.append({"site": s["name"], "metric": title, "rel": rel, "winner": who})
            md.append(f"| {meta[s['name']]['pretty']} | {title} | {np.nanmean(a):.3f} | "
                      f"{np.nanmean(b):.3f} | {rel:+.1f}% | {p:.2e} | {who} |")
    md.append("")
    fl = pd.DataFrame(order_flips)
    md.append("**一致性判定**：")
    for col in fl.metric.unique():
        sub = fl[fl.metric == col]
        consistent = sub.winner.nunique() == 1
        md.append(f"- `{col}`：{'✅ 三个站点一致' if consistent else '❌ 存在翻转'} "
                  f"（{', '.join(sub.winner.tolist())}）")
    md.append("")

    # R3
    md.append("## 5. R3 前瞻是否跨站点都不起作用（$B_2^{p}$ vs $B_2$）\n")
    md.append("| 站点 | $G(B_2)$ | $G(B_2^{p})$ | 变化 | 单步耗时倍数 |")
    md.append("|:--|--:|--:|--:|--:|")
    for s in SITES:
        def g(algo, col):
            sub = summ[(summ.site == s["name"]) & (summ.algorithm == algo)]
            return float(sub[col + "_mean"].iloc[0]) if not sub.empty else np.nan
        gb2, gb2p = g("B2_GREEDY", "aperture_gain"), g("B2_GREEDY_PREV__w2", "aperture_gain")
        sb2, sb2p = g("B2_GREEDY", "average_single_step_ms"), g("B2_GREEDY_PREV__w2", "average_single_step_ms")
        chg = 100 * (gb2p - gb2) / gb2 if np.isfinite(gb2) and gb2 > 0 else np.nan
        md.append(f"| {meta[s['name']]['pretty']} | {gb2:.3f} | {gb2p:.3f} | "
                  f"{chg:+.1f}% | {sb2p/sb2:.1f}× |")
    md.append("")

    # R4
    md.append("## 6. R4 连通度是否仍不构成区分判据\n")
    md.append("| 站点 | λ₂ 最小控制器 | λ₂ 最大控制器 | λ₂ 全距 | 链路可用率范围 |")
    md.append("|:--|:--|:--|--:|:--|")
    for s in SITES:
        sub = summ[summ.site == s["name"]]
        if sub.empty:
            continue
        lo = sub.loc[sub["average_connectivity_mean"].idxmin()]
        hi = sub.loc[sub["average_connectivity_mean"].idxmax()]
        la = sub["link_availability_ratio_mean"]
        md.append(f"| {meta[s['name']]['pretty']} | {LABEL.get(lo.algorithm)} "
                  f"({lo['average_connectivity_mean']:.2f}) | {LABEL.get(hi.algorithm)} "
                  f"({hi['average_connectivity_mean']:.2f}) | "
                  f"{hi['average_connectivity_mean']-lo['average_connectivity_mean']:.2f} | "
                  f"{la.min():.3f} – {la.max():.3f} |")
    md.append("")

    # R5
    if ref is not None:
        md.append("## 7. R5 参考次优是否跨站点存在（纯几何量）\n")
        md.append("$v_{\\mathrm{ref}}$（速率受限参考）与瞬时最优 $v_{\\mathrm{demand}}$ 的"
                  "**无向夹角**（0–90°），按每个站点实际使用的 20 个窗口统计。\n")
        md.append("| 站点 | 窗口数 | 平均 | 中位 | P90 | 最大 | >30° 占比 | >45° 占比 |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
        for _, r in ref.iterrows():
            md.append(f"| {meta[r['site']]['pretty']} | {int(r['n_windows_used'])} | "
                      f"{r['angle_mean_deg']:.1f}° | {r['angle_median_deg']:.1f}° | "
                      f"{r['angle_p90_deg']:.1f}° | {r['angle_max_deg']:.1f}° | "
                      f"{100*r['frac_gt30']:.1f}% | {100*r['frac_gt45']:.1f}% |")
        md.append("")

    md.append("## 8. 站点几何差异（对照）\n")
    md.append("| 站点 | 仰角均值 (°) | 窗口有效占比 | 星下点均距 (km) | 实测跨度均值 (m) |")
    md.append("|:--|--:|--:|--:|--:|")
    for s in SITES:
        sub = summ[summ.site == s["name"]]
        if sub.empty:
            continue
        def _m(col: str) -> str:
            if col not in sub.columns:
                return "—"
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            return f"{v.mean():.1f}" if len(v) else "—"
        md.append(f"| {meta[s['name']]['pretty']} | {_m('elev_mean_deg_mean')} | "
                  f"{_m('task_valid_fraction_mean')} | {_m('subpoint_dist_mean_km_mean')} | "
                  f"{_m('aperture_span_m_mean')} |")
    md.append("")

    out = HERE.parent / "多站点鲁棒性检验_报告.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[analyze] wrote {out}")
    print(f"[analyze] wrote {HERE / 'multisite_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
