from __future__ import annotations

from pathlib import Path

from tle_swarm_sim.config import PipelineConfig
from tle_swarm_sim.dataset import build_pass_samples
from tle_swarm_sim.tle import load_tle_records


SAMPLE_TLE = """STARLINK-1000
1 44235U 19029B   24180.00000000  .00002182  00000-0  10270-3 0  9991
2 44235  53.0000 100.0000 0001000  90.0000 270.0000 15.05500000    01
STARLINK-1001
1 44236U 19029C   24180.10000000  .00002182  00000-0  10270-3 0  9999
2 44236  53.0000 100.5000 0001000  91.0000 271.0000 15.05500000    02
"""


def test_load_tle_records(tmp_path: Path):
    path = tmp_path / "starlink.tle"
    path.write_text(SAMPLE_TLE, encoding="utf-8")
    records = load_tle_records(path)
    assert len(records) == 2
    assert records[0].catalog_number == "44235"


def test_build_pass_samples_smoke(tmp_path: Path):
    path = tmp_path / "starlink.tle"
    path.write_text(SAMPLE_TLE, encoding="utf-8")
    records = load_tle_records(path)
    config = PipelineConfig(
        repo_root=tmp_path,
        raw_dir=Path("raw"),
        processed_dir=Path("processed"),
        reports_dir=Path("reports"),
        target_samples=1,
        sample_minutes=8,
        sample_step_seconds=30,
    ).resolve()
    samples = build_pass_samples(records, config)
    assert isinstance(samples, list)

