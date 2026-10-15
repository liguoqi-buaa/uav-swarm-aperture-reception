"""合并论文主表 trial 与独立 baseline trial，做配对比较。

数据源
------
A) 论文 matched 主表：reports/scenario_c_matched/raw_trials/scenario_c_trials.csv
   （含 P0-P3 与 ST-GNN-DMPC，由另一 agent 产出）
B) 独立 baseline：reports/scenario_c_independent_matched_full/raw_trials/scenario_c_trials.csv
   （B1_GEOM_PD / B2_GREEDY / 各自 _H5）

两组 seed 相同（_trial_seed 只依赖 trial），故可按 trial_id 精确配对。

只读；输出 combined_comparison.csv 与屏幕报告。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / "uav_swarm_scenario_C"
PAPER = ROOT / "reports" / "scenario_c_matched" / "raw_trials" / "scenario_c_trials.csv"
INDEP = ROOT / "reports" / "scenario_c_independent_matched_full" / "raw_trials" / "scenario_c_trials.csv"
OUT = Path(__file__).resolve().parent / "combined_comparison.csv"

RENAME = {
    "OLSR-FANET": "P0", "APF-TC": "P1", "GNN-TC": "P2", "DMPC-C": "P3",
    "ST-GNN-DMPC": "ST-GNN-DMPC",
    "B1_GEOM_PD": "B1_GEOM_PD", "B2_GREEDY": "B2_GREEDY",
    "B1_GEOM_PD_H5": "B1_GEOM_PD_H5", "B2_GREEDY_H5": "B2_GREEDY_H5",
}
ORDER = ["P0", "P1", "P2", "P3", "B1_GEOM_PD", "B1_GEOM_PD_H5",
         "B2_GREEDY", "B2_GREEDY_H5", "ST-GNN-DMPC"]

# 归一化对齐质量（等长度）由 normalize_alignment.py 产出，此处直接写死其 20-trial 均值
A_LEN = {
    "P0": 0.273, "P1": 0.283, "P2": 0.290, "P3": 0.298,
    "ST-GNN-DMPC": 0.524,
    "B1_GEOM_PD": 0.562, "B1_GEOM_PD_H5": 0.566,
    "B2_GREEDY": 0.615, "B2_GREEDY_H5": 0.627,
}
A_DIR = {
    "P0": 0.330, "P1": 0.340, "P2": 0.347, "P3": 0.357,
    "ST-GNN-DMPC": 0.553,
    "B1_GEOM_PD": 0.652, "B1_GEOM_PD_H5": 0.656,
    "B2_GREEDY": 0.710, "B2_GREEDY_H5": 0.723,
}

KEY = ["aperture_gain", "aperture_alignment", "aperture_misalignment_deg",
       "aperture_span_m", "aperture_coverage", "aperture_wavefront_rms_m",
       "average_connectivity", "link_availability_ratio", "average_single_step_ms"]


def load() -> pd.DataFrame:
    a = pd.read_csv(PAPER)
    b = pd.read_csv(INDEP)
    df = pd.concat([a, b], ignore_index=True)
    df["algo"] = df["algorithm"].map(RENAME).fillna(df["algorithm"])
    df.to_csv(OUT, index=False)
    return df


def main() -> int:
    df = load()
    present = [a for a in ORDER if a in set(df["algo"])]
    print("=" * 118)
    print("合并比较（matched, n=20 trials/方法，按 trial_id 配对）")
    print("=" * 118)
    hdr = f"{'algorithm':<16}" + "".join(f"{k[:13]:>14}" for k in KEY[:5]) + f"{'A@3000':>10}{'Adir@3000':>11}"
    print(hdr)
    print("-" * len(hdr))
    for a in present:
        s = df[df["algo"] == a]
        row = f"{a:<16}"
        for k in KEY[:5]:
            v = pd.to_numeric(s[k], errors="coerce")
            row += f"{v.mean():>14.4f}"
        row += f"{A_LEN.get(a, float('nan')):>10.3f}{A_DIR.get(a, float('nan')):>11.3f}"
        print(row)

    print()
    print("=" * 118)
    print("配对检验：ST-GNN-DMPC vs 各方法")
    print("=" * 118)
    st = df[df["algo"] == "ST-GNN-DMPC"].sort_values("trial_id")
    for a in present:
        if a == "ST-GNN-DMPC":
            continue
        o = df[df["algo"] == a].sort_values("trial_id")
        n = min(len(st), len(o))
        out = [f"  vs {a:<14}"]
        for k in ["aperture_gain", "aperture_alignment", "average_connectivity", "aperture_span_m"]:
            x = pd.to_numeric(st[k], errors="coerce").to_numpy()[:n]
            y = pd.to_numeric(o[k], errors="coerce").to_numpy()[:n]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                out.append(f"{k[:9]}: n/a")
                continue
            d = x[m] - y[m]
            rel = 100.0 * d.mean() / max(abs(y[m].mean()), 1e-12)
            try:
                _t, p = stats.ttest_rel(x[m], y[m])
            except Exception:
                p = float("nan")
            out.append(f"{k[:9]}: {rel:+7.1f}% (p={p:.1e})")
        print(" | ".join(out))

    print()
    print("=== 主旨结论 ===")
    g = df.groupby("algo")["aperture_gain"].mean()
    best = g.drop(index=["ST-GNN-DMPC"], errors="ignore").idxmax()
    print(f"  最高 aperture gain: {best} = {g[best]:.4f}")
    print(f"  ST-GNN-DMPC       = {g['ST-GNN-DMPC']:.4f}")
    print(f"  → {best} 是 ST-GNN-DMPC 的 {g[best]/g['ST-GNN-DMPC']:.2f} 倍")
    print()
    print(f"  等长度对齐 A@3000:  最高 = {max(A_LEN.values()):.3f}"
          f"（{max(A_LEN, key=A_LEN.get)}）；ST-GNN-DMPC = {A_LEN['ST-GNN-DMPC']:.3f}")
    print(f"  纯朝向 Adir@3000:   最高 = {max(A_DIR.values()):.3f}"
          f"（{max(A_DIR, key=A_DIR.get)}）；ST-GNN-DMPC = {A_DIR['ST-GNN-DMPC']:.3f}")
    print()
    conn = df.groupby("algo")["average_connectivity"].mean()
    print(f"  连通度 λ2：ST-GNN-DMPC = {conn['ST-GNN-DMPC']:.2f}，"
          f"最佳独立 baseline = {conn[[a for a in present if a.startswith('B')]].max():.2f}")
    ms = df.groupby("algo")["average_single_step_ms"].mean()
    print(f"  单步耗时：ST-GNN-DMPC = {ms['ST-GNN-DMPC']:.2f} ms，"
          f"B2_GREEDY = {ms['B2_GREEDY']:.3f} ms  → 快 {ms['ST-GNN-DMPC']/ms['B2_GREEDY']:.0f} 倍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
