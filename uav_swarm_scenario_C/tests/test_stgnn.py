from __future__ import annotations

import torch

from tle_swarm_sim.simulation.stgnn import PhysicsInformedSTGNN, model_size_mb


def test_stgnn_forward_shape():
    model = PhysicsInformedSTGNN(node_feature_dim=12, physics_prior_dim=9, hidden_dim=16, prediction_horizon=5)
    node_features = torch.randn(2, 10, 6, 12)
    adjacency = torch.ones(2, 10, 6, 6)
    physics_prior = torch.randn(2, 9)
    output = model(node_features, adjacency, physics_prior)
    assert output.shape == (2, 5, 6, 3)
    assert model_size_mb(model) < 5.0

