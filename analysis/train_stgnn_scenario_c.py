"""Scenario C ST-GNN 重训脚本。

在物理自洽的分布式孔径场景（scenario_c）上重建"势场梯度"训练数据集并重训 ST-GNN，
替代在旧缩放场景上训练的权重，把论文里的"跨场景迁移性能"升级为"场景 C 的策略上限"。

与原 `scripts/train_stgnn_closed_loop.py` 的唯一实质差异：
  1. 数据生成改用场景 C 的孔径任务（`build_aperture_task`），而非旧场景的
     `build_satellite_task`（后者有隐性缩放，物理不自洽）；
  2. 势场梯度标签的参考点因此自动变成"参考孔径直线槽位"（`task.positions_m`
     在场景 C 下即为逐机槽位，`monitoring_slots` 走恒等指派分支）；
  3. 权重保存到新路径，不覆盖旧权重（旧权重是"迁移性能"的对比基准）。

用法（在 `analysis` 目录下，用 venv 解释器）：
    python train_stgnn_scenario_c.py --samples 900 --epochs 18 --batch-size 32 \
        --output models/stgnn/best_model_scenario_c.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1] / "uav_swarm_scenario_C"
sys.path.insert(0, str(ROOT))

from src.controllers.feature_builder import NODE_FEATURE_DIM, build_stgnn_features  # noqa: E402
from src.models.stgnn_model import ClosedLoopSTGNN, save_checkpoint  # noqa: E402
from tle_swarm_sim.simulation.config import SimulationConfig  # noqa: E402
from tle_swarm_sim.simulation.environment import initial_swarm_state  # noqa: E402
from tle_swarm_sim.simulation.experiment import _load_records  # noqa: E402
from tle_swarm_sim.simulation.potential import composite_potential  # noqa: E402
from tle_swarm_sim.simulation.scenario_c import (  # noqa: E402
    build_aperture_task,
    group_records_for_window,
    select_aperture_window,
)

POOL = Path(__file__).resolve().parent / "covisible_pool.json"

# ST-GNN-DMPC 的势场增益（对应 controllers.ALGORITHM_PROFILES["ST-GNN-DMPC"]
# 的 attr/rep/conn/power 四个 scale），用于生成"教师"梯度标签，与闭环推理一致。
GAINS_SCALE = (1.15, 1.0, 1.25, 0.45)


def build_config(out_dir: Path) -> SimulationConfig:
    """与 run_scenario_c.py 的孔径场景参数保持一致。"""
    return SimulationConfig(
        repo_root=ROOT,
        reports_dir=out_dir / "reports",
        figures_dir=out_dir / "figures",
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
        report_tag="train",
    ).resolve()


def _build_aperture_task(records, config: SimulationConfig, sample_idx: int):
    """按场景 C 逻辑构建孔径任务：选共视窗口 -> 取 3 星 -> 构建参考孔径槽位。"""
    window = select_aperture_window(records, config, sample_idx)
    group = group_records_for_window(records, window)
    start_time = datetime.fromisoformat(window["start_utc"])
    task, _geo = build_aperture_task(group, start_time, config)
    return task


def make_dataset(config: SimulationConfig, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    records = _load_records(config)
    xs, adjs, priors, ys = [], [], [], []
    for idx in range(samples):
        task = _build_aperture_task(records, config, idx)
        positions, velocities = initial_swarm_state(task, config, rng)
        # 与旧训练脚本一致：80% 概率在初始撒点附近做小扰动，增加样本多样性
        if rng.random() < 0.8:
            positions[:, :2] += rng.normal(
                0.0, config.initial_position_spread_m * 0.35, size=(config.num_uavs, 2)
            )
        max_step = max(0, task.positions_m.shape[0] - config.prediction_horizon - 1)
        step = int(rng.integers(0, max_step + 1)) if max_step > 0 else 0
        batch = build_stgnn_features(positions, velocities, task, step, config)
        label = np.zeros((config.prediction_horizon, config.num_uavs, 3), dtype=np.float32)
        for h in range(config.prediction_horizon):
            h_step = min(task.positions_m.shape[0] - 1, step + h)
            result = composite_potential(
                positions, velocities, task, h_step, config, gains_scale=GAINS_SCALE
            )
            label[h] = (result.gradient / 1000.0).astype(np.float32)
        xs.append(batch.node_features)
        adjs.append(batch.adjacency)
        priors.append(batch.physics_prior)
        ys.append(label)
        if (idx + 1) % 100 == 0:
            print(f"  [data] {idx + 1}/{samples}", flush=True)
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(adjs), dtype=torch.float32),
        torch.tensor(np.stack(priors), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=202604)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--output", type=str, default="models/stgnn/best_model_scenario_c.pt")
    args = ap.parse_args()

    out_dir = ROOT / "reports" / "stgnn_scenario_c"
    config = build_config(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] building dataset: {args.samples} samples (seed={args.seed})", flush=True)
    node_features, adjacency, physics_prior, labels = make_dataset(config, args.samples, args.seed)
    print(f"[train] dataset ready: {node_features.shape}", flush=True)

    train_count = int(0.8 * args.samples)
    train_ds = TensorDataset(node_features[:train_count], adjacency[:train_count],
                             physics_prior[:train_count], labels[:train_count])
    test_ds = TensorDataset(node_features[train_count:], adjacency[train_count:],
                            physics_prior[train_count:], labels[train_count:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = ClosedLoopSTGNN(node_feature_dim=NODE_FEATURE_DIM, physics_prior_dim=16,
                            hidden_dim=64, prediction_horizon=config.prediction_horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    rows = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, ab, pb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, ab, pb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        test_losses = []
        with torch.no_grad():
            for xb, ab, pb, yb in test_loader:
                test_losses.append(float(loss_fn(model(xb, ab, pb), yb).item()))
        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))
        rows.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"  [epoch {epoch:2d}] train={train_loss:.6f} test={test_loss:.6f}", flush=True)
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = ROOT / args.output
    metadata = {
        "node_feature_dim": NODE_FEATURE_DIM,
        "physics_prior_dim": 16,
        "hidden_dim": 64,
        "prediction_horizon": config.prediction_horizon,
        "gradient_output_scale": 1000.0,
        "training_status": "trained_scenario_c_aperture_potential_gradient",
        "scenario": "aperture",
        "train_samples": train_count,
        "test_samples": args.samples - train_count,
        "epochs": args.epochs,
        "best_test_loss": best_loss,
    }
    save_checkpoint(model, out_path, metadata)
    print(f"[train] saved model to {out_path}", flush=True)

    # 梯度预测误差（供论文/对比）
    model.eval()
    with torch.no_grad():
        pred = model(node_features[train_count:], adjacency[train_count:], physics_prior[train_count:])
    err = (pred - labels[train_count:]).numpy() * 1000.0
    target = labels[train_count:].numpy() * 1000.0
    grad_df = pd.DataFrame([{
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "naive_mae_m": float(np.mean(np.abs(target))),
        "naive_rmse_m": float(np.sqrt(np.mean(target ** 2))),
    }])
    grad_df.to_csv(out_dir / "gradient_prediction_error.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "training_log.csv", index=False)
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters={param_count} best_test_loss={best_loss:.6f}", flush=True)
    print(f"[train] gradient MAE={grad_df['mae_m'][0]:.1f} m  (naive={grad_df['naive_mae_m'][0]:.1f} m)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
