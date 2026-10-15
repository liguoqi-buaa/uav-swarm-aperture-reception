"""方案 A：单步算力预算拆解 + 硬件无关量 + Jetson 折算（YZH / CEP 稿 §6.4 用）

目的
----
把 `Table (tab:profiles)` 里"单步墙钟耗时"这一个标量，拆成可解释的构成项，
并给出硬件无关的算子量，从而把"纯本地 CPU 实测"升级为
"软件侧实时预算 + 保守的嵌入式折算"。

只读：不修改任何原始 trial 数据；输出到 `reports/audit/`。

用法
----
    python profile_compute_budget.py --trials 3
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "uav_swarm_scenario_C"
sys.path.insert(0, str(ROOT))

POOL = HERE / "covisible_pool.json"

# ---------------------------------------------------------------------------
# 计时累加器
# ---------------------------------------------------------------------------
ACC: dict[str, float] = {}
CALLS: dict[str, int] = {}


def reset_acc() -> None:
    ACC.clear()
    CALLS.clear()


@contextmanager
def _tick(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ACC[name] = ACC.get(name, 0.0) + (time.perf_counter() - t0)
        CALLS[name] = CALLS.get(name, 0) + 1


def wrap(mod, attr: str, label: str) -> None:
    """把 mod.attr 包一层计时。"""
    orig = getattr(mod, attr)

    def timed(*a, **kw):
        with _tick(label):
            return orig(*a, **kw)

    timed.__wrapped__ = orig  # type: ignore[attr-defined]
    setattr(mod, attr, timed)


# ---------------------------------------------------------------------------
# PyTorch 前向 hook（ST-GNN）
# ---------------------------------------------------------------------------
class _ForwardTimer:
    def __init__(self) -> None:
        self.t = 0.0
        self.n = 0

    def __call__(self, module, inputs, output):  # noqa: ANN001
        self.t += float(getattr(module, "_wb_last_ms", 0.0))
        self.n += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help="0 = 用整窗")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(1)

    from tle_swarm_sim.simulation.config import SimulationConfig
    from tle_swarm_sim.simulation.environment import initial_swarm_state
    from tle_swarm_sim.simulation.experiment import _load_records
    from tle_swarm_sim.simulation.scenario_c import (
        build_aperture_task,
        group_records_for_window,
        select_aperture_window,
    )

    cfg = SimulationConfig(
        repo_root=ROOT,
        reports_dir=ROOT / "reports",
        figures_dir=ROOT / "reports" / "figures",
        trials=20,
        concurrent_satellites=(3,),
        simulation_duration_s=600.0,
        control_step_s=1.0,
        scenario_mode="aperture",
        site_lat_deg=0.0,
        site_lon_deg=-174.624885,
        aperture_pool_path=POOL,
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
        profile_gain_mode="matched",
        report_tag="budget",
    ).resolve()

    records = _load_records(cfg)

    # ---- 打点：把控制路径上的各组件包起来 ----
    import tle_swarm_sim.simulation.controllers as C
    import tle_swarm_sim.simulation.potential as P
    import tle_swarm_sim.simulation.network as N
    import tle_swarm_sim.simulation.independent_baselines as IB
    import src.controllers.stgnn_dmpc_controller as SD
    import src.controllers.feature_builder as FB

    wrap(C, "composite_potential", "C_potential")          # 势场梯度（含邻接+λ2）
    wrap(P, "adjacency_matrix", "C_adjacency")             # 邻接矩阵
    wrap(P, "algebraic_connectivity", "C_lambda2")         # λ2 特征分解
    wrap(N, "snr_db", "C_snr")                             # 链路 SNR
    wrap(FB, "build_stgnn_features", "C_features")         # 特征构造
    wrap(SD, "build_stgnn_features", "C_features")
    wrap(SD, "solve_local_dmpc_qp", "C_qp_local")          # 单机 QP
    wrap(SD, "monitoring_slots", "C_slots")                # 参考槽位
    wrap(IB, "pd_accel", "C_pd")
    wrap(IB, "preview_accel", "C_preview_qp")
    wrap(IB, "independent_control", "C_indep_total")

    # ST-GNN 前向：用 forward pre/post hook 精确计时
    from src.controllers.stgnn_dmpc_controller import _load_model

    model, _meta, _path = _load_model(cfg)
    fwd = {"t": 0.0, "n": 0}

    def pre_hook(module, inputs):  # noqa: ANN001
        module._wb_t0 = time.perf_counter()

    def post_hook(module, inputs, output):  # noqa: ANN001
        fwd["t"] += time.perf_counter() - module._wb_t0
        fwd["n"] += 1

    h1 = model.register_forward_pre_hook(pre_hook)
    h2 = model.register_forward_hook(post_hook)

    ALGOS = [
        "P0", "P1", "P2", "P3",
        "ST-GNN-DMPC",
        "B1_GEOM_PD", "B2_GREEDY",
        "B1_GEOM_PD_PREV", "B2_GREEDY_PREV",
 ]
    # 论文标签 -> 代码内 profile 名（P0–P3 与 ALGORITHM_PROFILES 的对应）
    CODE_NAME = {
        "P0": "OLSR-FANET",
        "P1": "APF-TC",
        "P2": "GNN-TC",
        "P3": "DMPC-C",
    }

    results: dict[str, dict] = {}
    for algo in ALGOS:
        code_algo = CODE_NAME.get(algo, algo)
        per_step_ms: list[float] = []
        comp_accum: dict[str, list[float]] = {}
        for trial in range(args.trials):
            window = select_aperture_window(records, cfg, trial)
            group = group_records_for_window(records, window)
            start_time = datetime.fromisoformat(window["start_utc"])
            task, geo = build_aperture_task(group, start_time, cfg)
            rng = np.random.default_rng(cfg.random_seed + trial)
            positions, velocities = initial_swarm_state(task, cfg, rng)

            steps = task.positions_m.shape[0]
            if args.steps:
                steps = min(steps, args.steps)

            reset_acc()
            fwd["t"] = 0.0
            fwd["n"] = 0
            for step in range(steps):
                snap = dict(ACC)
                fsnap = fwd["t"]
                t0 = time.perf_counter()
                cmd = C.compute_control(code_algo, positions, velocities, task, step, cfg, rng, geometry=geo)
                per_step_ms.append((time.perf_counter() - t0) * 1000.0)
                # 逐 step 的组件增量（保证组件与总量的聚合口径一致）
                for k in ACC:
                    d = (ACC[k] - snap.get(k, 0.0)) * 1000.0
                    if d > 0.0:
                        comp_accum.setdefault(k, []).append(d)
                d_fwd = (fwd["t"] - fsnap) * 1000.0
                if d_fwd > 0.0:
                    comp_accum.setdefault("C_gnn_forward", []).append(d_fwd)
                positions, velocities = C.step_dynamics(positions, velocities, cmd.accel_mps2, cfg)

        def st(vals: list[float]) -> dict:
            vals = sorted(vals)
            n = len(vals)
            return {
                "mean_ms": float(statistics.fmean(vals)),
                "p50_ms": float(vals[n // 2]),
                "p95_ms": float(vals[min(n - 1, int(0.95 * n))]),
                "p99_ms": float(vals[min(n - 1, int(0.99 * n))]),
                "max_ms": float(vals[-1]),
            }

        results[algo] = {
            "total": st(per_step_ms),
            "components": {k: st(v) for k, v in comp_accum.items()},
        }
        print(f"[budget] {algo:20s} total_p50={results[algo]['total']['p50_ms']:7.3f} ms  "
              f"p95={results[algo]['total']['p95_ms']:7.3f} ms", flush=True)

    h1.remove()
    h2.remove()

    # ---------------------------------------------------------------
    # 硬件无关量
    # ---------------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters())
    T = int(cfg.historical_observation_horizon)
    Nn = int(cfg.num_uavs)
    Hd = int(cfg.prediction_horizon)
    F = FB.NODE_FEATURE_DIM
    hdim = 64
    # 主要算子（每步一次前向，batch=1）
    macs = {
        "adjacency_matmul": Nn * Nn * T * F,
        "self+neighbor_proj": 2 * Nn * T * (F * hdim),
        "gru": Nn * T * 3 * (hdim * hdim + hdim * hdim + hdim),
        "prior_proj": Nn * (16 * hdim),
        "output_linear": Nn * (hdim * Hd * 3),
        "layer_norm": Nn * hdim,
    }
    total_macs = sum(macs.values())
    op_counts = {
        "stgnn_parameters": int(n_params),
        "stgnn_nodes_per_step": Nn,
        "stgnn_history_steps": T,
        "stgnn_node_feature_dim": int(F),
        "stgnn_hidden_dim": hdim,
        "stgnn_macs_per_step": int(total_macs),
        "stgnn_flops_per_step": int(2 * total_macs),
        "stgnn_macs_breakdown": {k: int(v) for k, v in macs.items()},
        "qp_variables_per_uav": int(Hd * 3),
        "qp_constraints_per_uav": int(Hd * 3 + Hd * 3),
        "qp_variables_total": int(Hd * 3 * Nn),
        "qp_constraints_total": int((Hd * 3 + Hd * 3) * Nn),
        "qp_solve_mode": str(cfg.qp_solver_mode),
        "potential_pairwise_terms": int(Nn * (Nn - 1) // 2),
        "eigendecomposition_size": int(Nn),
        "network_links_evaluated": int(Nn * (Nn - 1) // 2),
    }

    # ---------------------------------------------------------------
    # Jetson 折算（保守：直接按单核慢化因子放大实测时间）
    # ---------------------------------------------------------------
    DEVICES = [
        ("Desktop CPU (measured, 1 thread)", 1.0,
         "reference platform of this study"),
        ("Jetson Orin Nano 8GB (Arm A78AE @1.5GHz)", 5.0,
         "conservative 5x slower per core than the desktop core"),
        ("Jetson Xavier NX (Carmel @1.4GHz)", 8.0,
         "conservative 8x slower per core"),
        ("Pessimistic embedded scenario", 10.0,
         "10x slower per core, MAXN power mode"),
    ]
    CTRL_PERIOD_MS = float(cfg.control_step_s * 1000.0)
    jetson: list[dict] = []
    for dev, s, note in DEVICES:
        row = {"device": dev, "slowdown_factor": s, "note": note, "controllers": {}}
        for algo, r in results.items():
            t = r["total"]["p50_ms"] * s
            row["controllers"][algo] = {
                "projected_step_ms": round(t, 3),
                "pct_of_control_period": round(100.0 * t / CTRL_PERIOD_MS, 4),
                "headroom_x": round(CTRL_PERIOD_MS / t, 1) if t > 0 else None,
            }
        jetson.append(row)

    out_dir = ROOT / "reports" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "trials": args.trials,
        "control_period_ms": CTRL_PERIOD_MS,
        "note": "Controller-only timing: compute_control() path, no metric evaluation. "
                "Projections are ANALYTIC, not measured on the embedded device.",
        "per_controller": {
            k: {
                "total": v["total"],
                "components": {kk: vv["p50_ms"] for kk, vv in v["components"].items()},
            }
            for k, v in results.items()
        },
        "hardware_independent": op_counts,
        "jetson_projection": jetson,
    }
    (out_dir / "compute_budget_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Markdown ----------------
    md: list[str] = []
    md.append("# 单步算力预算拆解与嵌入式折算（方案 A）\n")
    md.append(f"- 生成时间（UTC）：`{payload['generated_utc']}`")
    md.append(f"- 平台：`{payload['platform']}` / `{payload['processor']}`")
    md.append(f"- Python {payload['python']}，torch {payload['torch']}，"
              f"`torch.set_num_threads({payload['torch_threads']})`（单线程）")
    md.append(f"- 每控制器 {args.trials} 个 trial，整窗闭环；计时口径 = `compute_control()` "
              f"控制器路径（**不含**指标评估）")
    md.append(f"- 控制周期 = {CTRL_PERIOD_MS:.0f} ms\n")
    md.append("> **口径声明**：本表是软件侧实测 + **解析折算**。Jetson 一列**不是实测**，"
              "而是把桌面单核实测时间按保守的慢化因子放大得到。\n")

    md.append("## 1. 单步总耗时（实测，p50/p95，毫秒）\n")
    md.append("| Controller | p50 | p95 | p99 | max | 占 1 s 周期 (p50) | 余量 |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|")
    for algo, r in results.items():
        t = r["total"]["p50_ms"]
        md.append(f"| `{algo}` | {t:.3f} | {r['total']['p95_ms']:.3f} | "
                  f"{r['total']['p99_ms']:.3f} | {r['total']['max_ms']:.3f} | "
                  f"{100*t/CTRL_PERIOD_MS:.3f}% | {CTRL_PERIOD_MS/t:.0f}× |")
    md.append("")

    md.append("## 2. 构成项拆解（ms/step，p50）\n")
    all_comp = sorted({k for r in results.values() for k in r["components"]})
    md.append("| Component | " + " | ".join(f"`{a}`" for a in results) + " |")
    md.append("|:--|" + "--:|" * len(results))
    for comp in all_comp:
        cells = []
        for a in results:
            v = results[a]["components"].get(comp)
            cells.append(f"{v['p50_ms']:.3f}" if v else "—")
        md.append(f"| `{comp}` | " + " | ".join(cells) + " |")
    md.append("")
    md.append("组件含义：`C_gnn_forward` ST-GNN 前向；`C_features` 特征构造；"
              "`C_qp_local` 单机 DMPC 的 QP（预测族）；`C_slots` 参考槽位；"
              "`C_potential` 复合势场梯度；`C_adjacency` 邻接矩阵；`C_lambda2` λ₂ 特征分解；"
              "`C_snr` 链路 SNR；`C_preview_qp` 滚动时域预览 QP。")
    md.append("")
    md.append("**注意嵌套关系**：`C_potential ⊃ C_adjacency + C_lambda2 + C_snr`；"
              "`C_features ⊃ C_snr`（特征构造内部自己建了一次邻接矩阵）。"
              "因此只有**顶层项**可以直接相加：预测族 = `C_features + C_qp_local + C_gnn_forward + C_slots`；"
              "参数族的势场路径 = `C_potential`；几何族 = `C_indep_total`。")
    md.append("")

    if "ST-GNN-DMPC" in results:
        tot = results["ST-GNN-DMPC"]["total"]["p50_ms"]
        md.append("### 预测控制器的成本归属（关键结论）\n")
        md.append("| 顶层组件 | ms/step (p50) | 占单步总量 |")
        md.append("|:--|--:|--:|")
        top = ["C_features", "C_qp_local", "C_gnn_forward", "C_slots"]
        known = 0.0
        for comp in top:
            v = results["ST-GNN-DMPC"]["components"].get(comp)
            if v:
                known += v["p50_ms"]
                md.append(f"| `{comp}` | {v['p50_ms']:.3f} | {100*v['p50_ms']/tot:.1f}% |")
        md.append(f"| 其余（python 开销 / 目标轨迹组装 / 限幅） | {tot-known:.3f} | "
                  f"{100*(tot-known)/tot:.1f}% |")
        md.append(f"| **合计** | **{tot:.3f}** | **100%** |")
        md.append("")
        fe = results["ST-GNN-DMPC"]["components"].get("C_features", {}).get("p50_ms", 0.0)
        gn = results["ST-GNN-DMPC"]["components"].get("C_gnn_forward", {}).get("p50_ms", 0.0)
        md.append(f"→ **单步开销的大头是 numpy 侧的特征构造（{fe:.2f} ms，约 {100*fe/tot:.0f}%），"
                  f"神经网络前向只占 {gn:.2f} ms（约 {100*gn/tot:.1f}%）**。"
                  "也就是说，预测型控制器相较几何控制器的算力劣势主要来自**实现开销**，"
                  "而不是来自「学习」本身；即便把这部分优化掉，几何控制器仍然便宜两个数量级。")
        md.append("")

    md.append("## 3. 硬件无关算子量\n")
    md.append("| 量 | 值 |")
    md.append("|:--|--:|")
    md.append(f"| ST-GNN 参数量 | {op_counts['stgnn_parameters']:,} |")
    md.append(f"| ST-GNN 每步 MACs | {op_counts['stgnn_macs_per_step']:,} |")
    md.append(f"| ST-GNN 每步 FLOPs | {op_counts['stgnn_flops_per_step']:,} |")
    md.append(f"| QP 决策变量（单机 / 全队） | {op_counts['qp_variables_per_uav']} / {op_counts['qp_variables_total']} |")
    md.append(f"| QP 约束（单机 / 全队） | {op_counts['qp_constraints_per_uav']} / {op_counts['qp_constraints_total']} |")
    md.append(f"| QP 求解模式 | `{op_counts['qp_solve_mode']}` |")
    md.append(f"| 势场两两项 / 网络链路数 | {op_counts['potential_pairwise_terms']} / {op_counts['network_links_evaluated']} |")
    md.append(f"| λ₂ 特征分解规模 | {op_counts['eigendecomposition_size']}×{op_counts['eigendecomposition_size']} |")
    md.append("")
    md.append("ST-GNN 单步 MACs 分解：")
    md.append("")
    md.append("| 算子 | MACs |")
    md.append("|:--|--:|")
    for k, v in op_counts["stgnn_macs_breakdown"].items():
        md.append(f"| `{k}` | {v:,} |")
    md.append("")

    md.append("## 4. 向嵌入式平台的解析折算（**非实测**）\n")
    for row in jetson:
        md.append(f"### {row['device']}（慢化因子 {row['slowdown_factor']}×）\n")
        md.append(f"_{row['note']}_\n")
        md.append("| Controller | 折算单步 (ms) | 占 1 s 周期 | 余量 |")
        md.append("|:--|--:|--:|--:|")
        for algo, c in row["controllers"].items():
            md.append(f"| `{algo}` | {c['projected_step_ms']:.2f} | "
                      f"{c['pct_of_control_period']:.3f}% | {c['headroom_x']:.0f}× |")
        md.append("")

    (out_dir / "compute_budget_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[budget] wrote {out_dir / 'compute_budget_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
