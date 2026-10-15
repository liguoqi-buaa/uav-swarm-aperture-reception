from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    tle_group: str = "starlink"
    tle_url: str = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    source_tle_path: Path | None = None
    download_if_missing: bool = True
    strict_year_filter: bool = True
    start_year: int = 2024
    end_year: int = 2025
    min_orbit_altitude_km: float = 520.0
    max_orbit_altitude_km: float = 580.0
    min_inclination_deg: float = 52.5
    max_inclination_deg: float = 53.5
    target_samples: int = 1200
    target_concurrent_samples: int = 300
    train_ratio: float = 0.8
    min_elevation_deg: float = 10.0
    default_site_lat_deg: float = 20.0
    default_site_lon_deg: float = 180.0
    default_site_alt_m: float = 0.0
    sample_minutes: int = 18
    sample_step_seconds: int = 10
    min_visible_minutes: float = 2.0
    concurrency_sizes: tuple[int, ...] = (2, 3, 5)

    def resolve(self) -> "PipelineConfig":
        self.raw_dir = (self.repo_root / self.raw_dir).resolve()
        self.processed_dir = (self.repo_root / self.processed_dir).resolve()
        self.reports_dir = (self.repo_root / self.reports_dir).resolve()
        if self.source_tle_path is not None and not self.source_tle_path.is_absolute():
            self.source_tle_path = (self.repo_root / self.source_tle_path).resolve()
        return self
