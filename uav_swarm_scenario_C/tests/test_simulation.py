from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from tle_swarm_sim.simulation.config import SimulationConfig
from tle_swarm_sim.simulation.experiment import run_simulation_suite


def test_simulation_suite_smoke(tmp_path: Path, monkeypatch):
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    figures = reports / "figures"
    processed.mkdir()
    shutil.copyfile(Path("tests/fixtures/starlink_sample.tle"), processed / "starlink_archive.tle")
    monkeypatch.chdir(tmp_path)

    summary = run_simulation_suite(
        SimulationConfig(
            repo_root=Path("."),
            processed_dir=processed,
            reports_dir=reports,
            figures_dir=figures,
            trials=2,
            num_uavs=5,
            simulation_duration_s=2,
        )
    )
    assert "single_monitoring" in summary
    assert "multi3_monitoring" in summary
    assert (reports / "paper_table_2_single_monitoring.md").exists()
    assert (reports / "paper_table_5_realtime.md").exists()
    trial_index = pd.read_csv(reports / "trials" / "trial_index.csv")
    result_path = Path(trial_index.iloc[0]["result_path"])
    assert not result_path.is_absolute()
    assert (tmp_path / result_path).exists()
