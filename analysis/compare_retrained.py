"""对比 ST-GNN-DMPC 在「旧权重(迁移性能)」与「重训权重(场景上限)」下的闭环结果。

读取：
  - 旧（迁移）：uav_swarm_scenario_C/reports/scenario_c_{matched,published}/raw_trials/scenario_c_trials.csv
  - 新（重训）：uav_swarm_scenario_C/reports/scenario_c_retrained_{matched,published}/raw_trials/scenario_c_trials.csv
只取 ST-GNN-DMPC 行，同 trial 配对，输出核心指标 old vs new 的 mean±std 与相对变化。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "uav_swarm_scenario_C"

# 核心指标 + 方向（越大越好 / 越小越好）
METRICS = [
    ("aperture_gain", "higher"),
    ("aperture_alignment", "higher"),
    ("aperture_efficiency", "higher"),
    ("aperture_misalignment_deg", "lower"),
    ("aperture_misalignment_tail_deg", "lower"),
    ("aperture_wavefront_rms_m", "lower"),
    ("aperture_span_m", "higher"),
    ("average_connectivity", "higher"),
    ("link_availability_ratio", "higher"),
    ("average_node_degree", "higher"),
    ("connected_graph_ratio", "higher"),
    ("topology_convergence_latency_ms", "lower"),
    ("average_single_step_ms", "lower"),
    ("inference_latency_ms", "lower"),
]


def _stgnn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["algorithm"] == "ST-GNN-DMPC"].sort_values("trial_id").reset_index(drop=True)


def _report(mode: str) -> None:
    old = _stgnn(ROOT / "reports" / f"scenario_c_{mode}" / "raw_trials" / "scenario_c_trials.csv")
    new = _stgnn(ROOT / "reports" / f"scenario_c_retrained_{mode}" / "raw_trials" / "scenario_c_trials.csv")
    print(f"\n===== {mode} 模式：ST-GNN-DMPC 旧权重(迁移) vs 新权重(重训) =====")
    print(f"{'指标':32s} {'旧 mean±std':>22s} {'新 mean±std':>22s} {'相对变化':>12s}")
    for metric, direction in METRICS:
        if metric not in old.columns or metric not in new.columns:
            continue
        o = pd.to_numeric(old[metric], errors="coerce")
        n = pd.to_numeric(new[metric], errors="coerce")
        om, osd = o.mean(), o.std(ddof=1)
        nm, nsd = n.mean(), n.std(ddof=1)
        if not np.isfinite(om) or not np.isfinite(nm):
            continue
        rel = (nm - om) / abs(om) * 100.0
        # 同 trial 配对显著性（简单配对 t 检验）
        paired = pd.DataFrame({"old": o.values, "new": n.values}).dropna()
        pval = float("nan")
        if len(paired) >= 3:
            d = paired["new"].values - paired["old"].values
            se = d.std(ddof=1) / np.sqrt(len(d))
            if se > 1e-12:
                t = d.mean() / se
                # 用 scipy 更准，这里用正态近似
                from scipy import stats as _st
                pval = float(_st.ttest_rel(paired["new"], paired["old"]).pvalue)
        arrow = "↑" if (direction == "higher") == (nm > om) else "↓"
        print(f"{metric:32s} {om:10.4f}±{osd:8.4f} {nm:10.4f}±{nsd:8.4f} {rel:+10.2f}% {arrow} p={pval:.3g}")


def main() -> int:
    _report("matched")
    _report("published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
