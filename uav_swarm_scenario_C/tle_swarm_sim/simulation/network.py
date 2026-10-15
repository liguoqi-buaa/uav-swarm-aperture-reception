from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig

C_LIGHT = 299_792_458.0


def snr_db(distance_m: float, config: SimulationConfig) -> float:
    d = max(distance_m, 1.0)
    wavelength = C_LIGHT / config.carrier_frequency_hz
    free_space_loss_db = -20.0 * math.log10(wavelength / (4.0 * math.pi))
    distance_loss_db = 10.0 * config.path_loss_exponent * math.log10(d)
    noise_dbm = config.noise_density_dbm_hz + 10.0 * math.log10(config.bandwidth_hz)
    rx_power_dbm = (
        config.transmit_power_dbm
        + 2.0 * config.antenna_gain_db
        - free_space_loss_db
        - distance_loss_db
        - config.additional_loss_db
    )
    return rx_power_dbm - noise_dbm


def adjacency_matrix(positions_m: np.ndarray, config: SimulationConfig, previous_adj: np.ndarray | None = None) -> np.ndarray:
    n = positions_m.shape[0]
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(positions_m[i] - positions_m[j]))
            threshold = config.connectivity_snr_threshold_db
            if previous_adj is not None and previous_adj[i, j] > 0.5:
                threshold -= config.link_hysteresis_margin_db
            link_on = dist <= config.max_communication_range_m and snr_db(dist, config) >= threshold
            if link_on:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def laplacian(adj: np.ndarray) -> np.ndarray:
    return np.diag(adj.sum(axis=1)) - adj


def algebraic_connectivity(adj: np.ndarray) -> float:
    if adj.shape[0] < 2:
        return 0.0
    eigvals = np.linalg.eigvalsh(laplacian(adj))
    return float(np.sort(eigvals)[1])


def average_node_degree(adj: np.ndarray) -> float:
    return float(adj.sum(axis=1).mean()) if adj.size else 0.0


def link_disruption_rate(previous_adj: np.ndarray | None, current_adj: np.ndarray) -> float:
    if previous_adj is None:
        return 0.0
    n = current_adj.shape[0]
    possible = n * (n - 1) / 2
    if possible == 0:
        return 0.0
    previous_upper = np.triu(previous_adj, k=1)
    current_upper = np.triu(current_adj, k=1)
    disruptions = np.logical_xor(previous_upper > 0.5, current_upper > 0.5).sum()
    return float(disruptions / possible)
