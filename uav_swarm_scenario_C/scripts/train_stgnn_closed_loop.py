from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controllers.feature_builder import NODE_FEATURE_DIM, build_stgnn_features, feature_sample_rows
from src.controllers.stgnn_dmpc_controller import write_feature_contract
from src.models.stgnn_model import ClosedLoopSTGNN, save_checkpoint
from tle_swarm_sim.sgp4_utils import ecef_to_geodetic, satellite_ecef_at
from tle_swarm_sim.simulation.config import SimulationConfig
from tle_swarm_sim.simulation.environment import build_satellite_task, initial_swarm_state
from tle_swarm_sim.simulation.experiment import _load_records, _select_group
from tle_swarm_sim.simulation.potential import composite_potential


def _build_task(records, group_size: int, sample_idx: int, rng: np.random.Generator, config: SimulationConfig):
    group = _select_group(records, group_size, sample_idx, rng)
    start_time = group[0].epoch_datetime.astimezone(timezone.utc)
    site_ecef = satellite_ecef_at(group[0], start_time)
    site_lat, site_lon, _ = ecef_to_geodetic(site_ecef)
    return build_satellite_task(group, start_time, site_lat, site_lon, config), group


def _make_dataset(config: SimulationConfig, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    records = _load_records(config)
    xs, adjs, priors, ys = [], [], [], []
    sample_batches = {}
    for idx in range(samples):
        group_size = int(rng.choice([1, 2, 3, 5], p=[0.35, 0.2, 0.25, 0.2]))
        task, _group = _build_task(records, group_size, idx, rng, config)
        positions, velocities = initial_swarm_state(task, config, rng)
        if rng.random() < 0.8:
            positions[:, :2] += rng.normal(0.0, config.initial_position_spread_m * 0.35, size=(config.num_uavs, 2))
        max_step = max(0, task.positions_m.shape[0] - config.prediction_horizon - 1)
        step = int(rng.integers(0, max_step + 1)) if max_step > 0 else 0
        batch = build_stgnn_features(positions, velocities, task, step, config)
        label = np.zeros((config.prediction_horizon, config.num_uavs, 3), dtype=np.float32)
        for h in range(config.prediction_horizon):
            h_step = min(task.positions_m.shape[0] - 1, step + h)
            result = composite_potential(
                positions,
                velocities,
                task,
                h_step,
                config,
                gains_scale=(1.15, 1.0, 1.25, 0.45),
            )
            label[h] = (result.gradient / 1000.0).astype(np.float32)
        xs.append(batch.node_features)
        adjs.append(batch.adjacency)
        priors.append(batch.physics_prior)
        ys.append(label)
        if group_size in (1, 3, 5) and group_size not in sample_batches:
            sample_batches[group_size] = (batch, step)
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(adjs), dtype=torch.float32),
        torch.tensor(np.stack(priors), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
        sample_batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=202604)
    args = parser.parse_args()

    config = SimulationConfig().resolve()
    reports_dir = config.reports_dir / "stgnn"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "debug").mkdir(parents=True, exist_ok=True)
    write_feature_contract(config.reports_dir / "debug" / "stgnn_feature_contract.md")

    node_features, adjacency, physics_prior, labels, sample_batches = _make_dataset(config, args.samples, args.seed)
    for m, (batch, step) in sample_batches.items():
        pd.DataFrame(feature_sample_rows(batch, step)).to_csv(config.reports_dir / "debug" / f"stgnn_feature_sample_M{m}.csv", index=False)

    train_count = int(0.8 * args.samples)
    train_ds = TensorDataset(node_features[:train_count], adjacency[:train_count], physics_prior[:train_count], labels[:train_count])
    test_ds = TensorDataset(node_features[train_count:], adjacency[train_count:], physics_prior[train_count:], labels[train_count:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = ClosedLoopSTGNN(node_feature_dim=NODE_FEATURE_DIM, physics_prior_dim=16, hidden_dim=64, prediction_horizon=config.prediction_horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
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
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "node_feature_dim": NODE_FEATURE_DIM,
        "physics_prior_dim": 16,
        "hidden_dim": 64,
        "prediction_horizon": config.prediction_horizon,
        "gradient_output_scale": 1000.0,
        "training_status": "trained_teacher_potential_gradient",
        "train_samples": train_count,
        "test_samples": args.samples - train_count,
    }
    save_checkpoint(model, config.stgnn_model_path, metadata)

    log_df = pd.DataFrame(rows)
    log_df.to_csv(reports_dir / "training_log.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train")
    plt.plot(log_df["epoch"], log_df["test_loss"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("SmoothL1 loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "training_curves.png", dpi=180)
    plt.close()

    model.eval()
    with torch.no_grad():
        pred = model(node_features[train_count:], adjacency[train_count:], physics_prior[train_count:])
    err = (pred - labels[train_count:]).numpy() * 1000.0
    target = labels[train_count:].numpy() * 1000.0
    naive = np.zeros_like(target)
    grad_df = pd.DataFrame(
        [
            {
                "mae_m": float(np.mean(np.abs(err))),
                "rmse_m": float(np.sqrt(np.mean(err * err))),
                "naive_mae_m": float(np.mean(np.abs(naive - target))),
                "naive_rmse_m": float(np.sqrt(np.mean((naive - target) ** 2))),
            }
        ]
    )
    grad_df.to_csv(reports_dir / "gradient_prediction_error.csv", index=False)

    param_count = sum(p.numel() for p in model.parameters())
    size_mb = config.stgnn_model_path.stat().st_size / (1024.0 * 1024.0)
    (reports_dir / "model_summary.txt").write_text(
        f"ClosedLoopSTGNN\nnode_feature_dim={NODE_FEATURE_DIM}\nphysics_prior_dim=16\n"
        f"hidden_dim=64\nprediction_horizon={config.prediction_horizon}\nparameters={param_count}\n"
        f"train_samples={train_count}\ntest_samples={args.samples - train_count}\n",
        encoding="utf-8",
    )
    (reports_dir / "model_size_report.md").write_text(
        f"# ST-GNN model size\n\nFP32 checkpoint: {size_mb:.3f} MB\nParameter count: {param_count}\n",
        encoding="utf-8",
    )
    latency_inputs = (node_features[:1], adjacency[:1], physics_prior[:1])
    for _ in range(5):
        _ = model(*latency_inputs)
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            t0 = __import__("time").perf_counter()
            _ = model(*latency_inputs)
            times.append((__import__("time").perf_counter() - t0) * 1000.0)
    (reports_dir / "inference_latency_report.md").write_text(
        f"# ST-GNN inference latency\n\n本机 CPU 单样本平均推理延迟: {np.mean(times):.3f} ms\n"
        "未在 Jetson Xavier NX 上实测。\n",
        encoding="utf-8",
    )
    (reports_dir / "quantization_report.md").write_text(
        "# Quantization report\n\n当前闭环复现使用 FP32 checkpoint。INT8 量化未接入主闭环实验。\n",
        encoding="utf-8",
    )
    (reports_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(config.stgnn_model_path), "best_test_loss": best_loss, "size_mb": size_mb}, ensure_ascii=False))


if __name__ == "__main__":
    main()
