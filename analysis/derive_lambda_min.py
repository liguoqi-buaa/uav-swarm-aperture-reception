"""O4 第一步：从一致性/同步时间预算推导代数连通度下限 λ_min。

理论
----
相位/状态参考在蜂群内传播的最慢模式对应 λ₂。离散一致性迭代
    x(k+1) = x(k) - alpha * L * x(k)
的最慢模态衰减因子为 (1 - alpha*λ₂)，取最优步长 alpha = 1/λ_max，则
    (1 - λ₂/λ_max)^N_iter ≤ epsilon
    =>  λ₂ ≥ λ_max * (1 - epsilon^(1/N_iter))

其中 epsilon 为相对一致性误差目标，N_iter 为一个控制周期内可用的通信轮数。
故 λ_min 完全由 (λ_max, epsilon, N_iter) 决定。

本脚本只读：从各方法的逐帧轨迹重建邻接矩阵（复用 simulation.network 的判定规则），
统计 λ₂、λ_max 与谱比，再给出 λ_min(epsilon, N_iter) 网格。

用法：
    python derive_lambda_min.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "uav_swarm_scenario_C"
sys.path.insert(0, str(ROOT))

from tle_swarm_sim.simulation.config import SimulationConfig  # noqa: E402
from tle_swarm_sim.simulation.network import adjacency_matrix, algebraic_connectivity  # noqa: E402

REPORTS = ROOT / "reports"
CASES = [
    ("P3 (best profile)", REPORTS / "scenario_c_matched/trials/M3/DMPC-C", "aperture_M3_matched_trial_{:02d}"),
    ("ST blend=0.65", REPORTS / "scenario_c_o1_blend0.65_matched/trials/M3/ST-GNN-DMPC", "aperture_M3_o1b0.65_trial_{:02d}"),
    ("ST blend=0.4", REPORTS / "scenario_c_o1_blend0.4_matched/trials/M3/ST-GNN-DMPC", "aperture_M3_o1b0.4_trial_{:02d}"),
    ("ST blend=0.0", REPORTS / "scenario_c_o1_blend0.0_matched/trials/M3/ST-GNN-DMPC", "aperture_M3_o1b0.0_trial_{:02d}"),
    ("B2_GREEDY", REPORTS / "scenario_c_independent_matched_full/trials/M3/B2_GREEDY", "aperture_M3_indep_matched_trial_{:02d}"),
]
N_TRIALS = 8


def build_config() -> SimulationConfig:
    return SimulationConfig(
        repo_root=ROOT, reports_dir=REPORTS / "_tmp", figures_dir=REPORTS / "_tmp",
        trials=20, concurrent_satellites=(3,), simulation_duration_s=600.0, control_step_s=1.0,
        scenario_mode="aperture", site_lat_deg=0.0, site_lon_deg=-174.624885,
        aperture_pool_path=Path(__file__).resolve().parent / "covisible_pool.json",
        aperture_baseline_m=3000.0, aperture_sigma_m=500.0, aperture_min_elevation_deg=10.0,
        monitoring_kernel_radius_m=1000.0, deployment_mode="monitor_area_near",
        monitor_area_initial_spread_m=1500.0, max_communication_range_m=2500.0,
        task_window_mode="extended_pass_10deg",
    ).resolve()


def spectrum_stats(positions: np.ndarray, config: SimulationConfig,
                   previous_adj: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
    """返回 (λ₂, λ_max, 当前邻接矩阵)。positions 为 (N,3)。

    注意：必须传入 previous_adj，才能复现代码里的链路滞回（link_hysteresis_margin_db）；
    不传会低估连通度。
    """
    adj = adjacency_matrix(positions, config, previous_adj)
    lam2 = algebraic_connectivity(adj)
    a = np.asarray(adj, dtype=float)
    a = 0.5 * (a + a.T)
    deg = a.sum(axis=1)
    L = np.diag(deg) - a
    ev = np.linalg.eigvalsh(L)
    return float(lam2), float(ev[-1]), a


def main() -> int:
    cfg = build_config()
    print("=" * 100)
    print("各方法的网络谱（comm range 2500 m；逐帧重建邻接矩阵）")
    print("=" * 100)
    print(f"{'method':<20}{'λ2 (mean)':>12}{'λ2 (p05)':>12}{'λ_max (mean)':>15}{'λ2/λ_max':>12}")
    print("-" * 100)

    summary: dict[str, tuple[float, float, float]] = {}
    for name, root, pat in CASES:
        lam2s, lamaxs = [], []
        for t in range(N_TRIALS):
            f = root / pat.format(t) / "trajectory_uav.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            prev = None
            steps = np.sort(df["step"].unique())
            n = df["uav_id"].nunique()
            pos = (df.sort_values(["step", "uav_id"])[["x_m", "y_m", "z_m"]]
                     .to_numpy(dtype=float).reshape(len(steps), n, 3))
            for i in range(len(steps)):
                l2, lmax, prev = spectrum_stats(pos[i], cfg, prev)
                if i % 4 == 0:
                    lam2s.append(l2)
                    lamaxs.append(lmax)
        if not lam2s:
            print(f"{name:<20} 无数据")
            continue
        lam2s, lamaxs = np.array(lam2s), np.array(lamaxs)
        summary[name] = (float(lam2s.mean()), float(lamaxs.mean()), float(lam2s.mean() / lamaxs.mean()))
        print(f"{name:<20}{lam2s.mean():>12.3f}{np.percentile(lam2s, 5):>12.3f}"
              f"{lamaxs.mean():>15.2f}{lam2s.mean()/lamaxs.mean():>12.4f}")

    print()
    print("=" * 100)
    print("推导的 λ_min = λ_max · (1 − ε^(1/N_iter))")
    print("=" * 100)
    lmax_typical = np.mean([v[1] for v in summary.values()])
    print(f"取 λ_max = {lmax_typical:.2f}（各方法均值）")
    print()
    eps_grid = [0.20, 0.10, 0.05, 0.02, 0.01]
    n_grid = [5, 10, 20, 50, 100]
    hdr = f"{'ε \\ N_iter':<12}" + "".join(f"{n:>10}" for n in n_grid)
    print(hdr)
    print("-" * len(hdr))
    for eps in eps_grid:
        row = f"{eps:<12.2f}"
        for n in n_grid:
            row += f"{lmax_typical * (1.0 - eps ** (1.0 / n)):>10.2f}"
        print(row)

    print()
    print("=" * 100)
    print("判决：λ_min 落在哪一侧？")
    print("=" * 100)
    b2 = summary.get("B2_GREEDY")
    if b2:
        print(f"  B2_GREEDY 的 λ₂ = {b2[0]:.2f}")
        thresh_needed = b2[0] / lmax_typical
        print(f"  要排除 B2，需满足 (1 − ε^(1/N_iter)) > {thresh_needed:.4f}"
              f"  ⇔  (1 − λ₂/λ_max) = {1 - thresh_needed:.4f}")
        # 反推：给定 ε，需要多少轮才排除 B2
        for eps in [0.05, 0.10, 0.20]:
            need = eps ** (1.0 / (1.0 - thresh_needed))
            # 解 N: ε^(1/N) = 1 - λ2/λmax  =>  1/N = ln(1-λ2/λmax)/ln(ε) => N = ln(ε)/ln(1-λ2/λmax)
            import math
            N = math.log(eps) / math.log(1.0 - thresh_needed)
            print(f"    ε={eps:.2f} → 需 N_iter < {N:.1f} 轮才排除 B2（轮数越多越容易达成一致，越难排除）")
    print()
    print("  参考：论文 config 自带的 lambda_threshold = 0.2（连通势场的激活阈值）")
    print("  参考：论文 Table 5 报告 P0-P3 的平均节点度 13.0、ST-GNN-DMPC 17.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
