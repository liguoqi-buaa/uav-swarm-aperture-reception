from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    figures_dir: Path = field(default_factory=lambda: Path("reports/figures"))

    random_seed: int = 202604
    trials: int = 20
    max_samples_per_scenario: int = 20

    # Paper Table 1: satellite and task parameters.
    orbital_height_m: float = 550_000.0
    min_elevation_deg: float = 10.0
    optimal_monitoring_radius_m: float = 5_000.0
    monitoring_kernel_radius_m: float = 2_300.0
    monitoring_kernel_exponent: float = 2.0
    concurrent_satellites: tuple[int, ...] = (1, 2, 3, 5)

    # Paper Table 1: UAV parameters.
    num_uavs: int = 20
    max_velocity_mps: float = 20.0
    max_acceleration_mps2: float = 5.0
    safe_distance_m: float = 50.0
    uav_altitude_m: float = 300.0
    sample_time_s: float = 0.1
    simulation_duration_s: float = 60.0
    control_step_s: float = 1.0

    # Paper Table 1: communication parameters.
    carrier_frequency_hz: float = 3.5e9
    transmit_power_dbm: float = 20.0
    bandwidth_hz: float = 100e6
    connectivity_snr_threshold_db: float = 10.0
    max_communication_range_m: float = 1_000.0
    antenna_gain_db: float = 3.0
    path_loss_exponent: float = 2.0
    additional_loss_db: float = 5.0
    noise_density_dbm_hz: float = -174.0

    # Paper Table 1: algorithm parameters.
    historical_observation_horizon: int = 10
    prediction_horizon: int = 5
    k_attr: float = 0.8
    k_rep: float = 1.2
    k_conn: float = 1.5
    k_power: float = 0.5
    lambda_threshold: float = 0.2
    rho: float = 3.0
    control_penalty: float = 0.35
    velocity_penalty: float = 0.01
    terminal_cost_weight: float = 14.0
    velocity_feedforward_weight: float = 0.55
    gradient_scale: float = 0.02
    lambda_connectivity: float = 0.4
    lambda_collision: float = 0.8
    task_allocation_weight: float = 7.5
    stgnn_formation_offset_scale: float = 1.0
    stgnn_connectivity_blend: float = 0.65
    stgnn_connectivity_blend_single_satellite: float | None = None
    qp_solver_mode: str = "fast_projected"
    link_hysteresis_margin_db: float = 2.0
    monitoring_threshold: float = 0.79
    stgnn_model_path: Path = field(default_factory=lambda: Path("models/stgnn/best_model.pt"))

    # Reproducible local-plane transform for tractable UAV kinematics.
    local_track_radius_fraction: float = 0.2
    formation_radius_m: float = 250.0
    monitoring_slot_radius_m: float = 250.0
    slot_smoothing_alpha: float = 0.75
    initial_position_spread_m: float = 900.0
    deployment_mode: str = "warm_start_near_popt"
    monitor_area_initial_spread_m: float = 1_500.0
    warm_start_initial_spread_m: float = 220.0

    # Optional extended-window replay/re-run mode. The default keeps the
    # original Phase4 behavior unchanged.
    task_window_mode: str = "phase4_local"
    task_window_search_csv: Path = field(default_factory=lambda: Path("reports/debug/task_window_extended_pass_search.csv"))
    task_window_main_elevation_deg: float = 10.0

    # ------------------------------------------------------------------
    # Scenario C: physically self-consistent fixed monitoring area with
    # pass-predictive distributed-aperture topology control.
    # `scenario_mode="legacy"` keeps the original (scaled sub-satellite track)
    # behaviour byte-for-byte; `scenario_mode="aperture"` switches to the
    # real-geometry aperture task implemented in scenario_c.py.
    # ------------------------------------------------------------------
    scenario_mode: str = "legacy"
    site_lat_deg: float = 0.0
    site_lon_deg: float = -174.624885
    aperture_pool_path: Path | None = None
    aperture_baseline_m: float = 3_000.0
    aperture_sigma_m: float = 500.0
    aperture_min_elevation_deg: float = 15.0
    # "published": keep the paper's per-algorithm gains.
    # "matched":   equalise every baseline's potential-field gains to ST-GNN-DMPC
    #              so that only prediction horizon / response delay differ.
    profile_gain_mode: str = "published"
    # Free-form tag appended to the trial directory name so that multiple
    # configurations can coexist inside one reports/ tree.
    report_tag: str = ""

    def resolve(self) -> "SimulationConfig":
        self.repo_root = self.repo_root.resolve()
        self.processed_dir = (self.repo_root / self.processed_dir).resolve()
        self.reports_dir = (self.repo_root / self.reports_dir).resolve()
        self.figures_dir = (self.repo_root / self.figures_dir).resolve()
        self.stgnn_model_path = (self.repo_root / self.stgnn_model_path).resolve()
        self.task_window_search_csv = (self.repo_root / self.task_window_search_csv).resolve()
        if self.aperture_pool_path is not None:
            self.aperture_pool_path = Path(self.aperture_pool_path)
            if not self.aperture_pool_path.is_absolute():
                self.aperture_pool_path = (self.repo_root / self.aperture_pool_path).resolve()
        return self
