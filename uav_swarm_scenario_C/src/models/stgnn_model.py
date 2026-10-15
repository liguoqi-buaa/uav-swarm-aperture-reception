from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch
from torch import nn


class ClosedLoopSTGNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 71,
        physics_prior_dim: int = 16,
        hidden_dim: int = 64,
        prediction_horizon: int = 5,
    ) -> None:
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.physics_prior_dim = physics_prior_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        self.self_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_proj = nn.Linear(physics_prior_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * 3),
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor, physics_prior: torch.Tensor) -> torch.Tensor:
        batch, time_steps, nodes, _features = node_features.shape
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor = torch.matmul(adjacency, node_features) / degree
        x = torch.relu(self.self_proj(node_features) + self.neighbor_proj(neighbor))
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, time_steps, -1)
        _, hidden = self.temporal(x)
        hidden = hidden[-1].reshape(batch, nodes, -1)
        hidden = hidden + torch.relu(self.prior_proj(physics_prior)).unsqueeze(1)
        out = self.output(hidden).reshape(batch, nodes, self.prediction_horizon, 3)
        return out.permute(0, 2, 1, 3).contiguous()


def save_checkpoint(model: ClosedLoopSTGNN, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_stgnn_model(path: Path, device: str = "cpu") -> tuple[ClosedLoopSTGNN, dict]:
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    model = ClosedLoopSTGNN(
            node_feature_dim=int(metadata.get("node_feature_dim", 71)),
        physics_prior_dim=int(metadata.get("physics_prior_dim", 16)),
        hidden_dim=int(metadata.get("hidden_dim", 64)),
        prediction_horizon=int(metadata.get("prediction_horizon", 5)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


def model_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)
