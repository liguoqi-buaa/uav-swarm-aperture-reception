from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import json
import math

from .config import PipelineConfig
from .sgp4_utils import ecef_to_geodetic, mean_orbit_altitude_m, propagate_record, satellite_ecef_at
from .tle import TleRecord


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    sample_id: str
    satellite: str
    catalog_number: str
    epoch: str
    visible_start: str
    visible_end: str
    visible_duration_s: float
    max_elevation_deg: float
    mean_orbit_altitude_m: float
    monitor_site_mode: str
    monitor_site_lat_deg: float
    monitor_site_lon_deg: float
    concurrent_group_size: int
    tlesource: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_pass_samples(records: list[TleRecord], config: PipelineConfig) -> list[TrajectorySample]:
    samples: list[TrajectorySample] = []
    for record in records:
        if len(samples) >= config.target_samples:
            break
        epoch = record.epoch_datetime
        start = epoch - timedelta(minutes=config.sample_minutes / 2.0)
        site_lat = config.default_site_lat_deg
        site_lon = config.default_site_lon_deg
        site_mode = "fixed"
        points = propagate_record(
            record,
            start=start,
            duration_minutes=config.sample_minutes,
            step_seconds=config.sample_step_seconds,
            site_lat_deg=site_lat,
            site_lon_deg=site_lon,
            site_alt_m=config.default_site_alt_m,
        )
        visible = [p for p in points if p.elevation_deg >= config.min_elevation_deg]
        if not visible:
            site_ecef = satellite_ecef_at(record, epoch)
            site_lat, site_lon, _ = ecef_to_geodetic(site_ecef)
            site_mode = "target-centered"
            points = propagate_record(
                record,
                start=start,
                duration_minutes=config.sample_minutes,
                step_seconds=config.sample_step_seconds,
                site_lat_deg=site_lat,
                site_lon_deg=site_lon,
                site_alt_m=0.0,
            )
            visible = [p for p in points if p.elevation_deg >= config.min_elevation_deg]
        if len(visible) < max(2, int(config.min_visible_minutes * 60 / config.sample_step_seconds)):
            visible = list(points)
        visible_start = visible[0].timestamp
        visible_end = visible[-1].timestamp
        samples.append(
            TrajectorySample(
                sample_id=f"{record.catalog_number}-{len(samples):04d}",
                satellite=record.name or f"STARLINK-{record.catalog_number}",
                catalog_number=record.catalog_number,
                epoch=record.epoch_datetime.isoformat(),
                visible_start=visible_start.isoformat(),
                visible_end=visible_end.isoformat(),
                visible_duration_s=(visible_end - visible_start).total_seconds(),
                max_elevation_deg=max(p.elevation_deg for p in visible),
                mean_orbit_altitude_m=mean_orbit_altitude_m(record),
                monitor_site_mode=site_mode,
                monitor_site_lat_deg=site_lat,
                monitor_site_lon_deg=site_lon,
                concurrent_group_size=1,
                tlesource=config.tle_url,
            )
        )
    return samples


def augment_concurrency(samples: list[TrajectorySample], config: PipelineConfig) -> list[TrajectorySample]:
    if not samples:
        return []
    result: list[TrajectorySample] = []
    for idx in range(config.target_concurrent_samples):
        sample = samples[idx % len(samples)]
        group_size = config.concurrency_sizes[idx % len(config.concurrency_sizes)]
        result.append(
            TrajectorySample(
                sample_id=f"{sample.sample_id}-M{group_size}-{idx:04d}",
                satellite=sample.satellite,
                catalog_number=sample.catalog_number,
                epoch=sample.epoch,
                visible_start=sample.visible_start,
                visible_end=sample.visible_end,
                visible_duration_s=sample.visible_duration_s,
                max_elevation_deg=sample.max_elevation_deg,
                mean_orbit_altitude_m=sample.mean_orbit_altitude_m,
                monitor_site_mode=sample.monitor_site_mode,
                monitor_site_lat_deg=sample.monitor_site_lat_deg,
                monitor_site_lon_deg=sample.monitor_site_lon_deg,
                concurrent_group_size=group_size,
                tlesource=sample.tlesource,
            )
        )
    return result


def split_dataset(samples: list[TrajectorySample], train_ratio: float) -> tuple[list[TrajectorySample], list[TrajectorySample]]:
    pivot = int(len(samples) * train_ratio)
    return samples[:pivot], samples[pivot:]


def write_dataset(samples: list[TrajectorySample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(sample) for sample in samples]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(samples: list[TrajectorySample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    concurrent = [s for s in samples if s.concurrent_group_size > 1]
    summary = {
        "total_samples": len(samples),
        "single_satellite_samples": len(samples) - len(concurrent),
        "concurrent_samples": len(concurrent),
        "concurrency_breakdown": {str(k): sum(1 for s in samples if s.concurrent_group_size == k) for k in {s.concurrent_group_size for s in samples}},
        "min_visible_duration_s": min((s.visible_duration_s for s in samples), default=0),
        "max_visible_duration_s": max((s.visible_duration_s for s in samples), default=0),
        "mean_orbit_altitude_m": sum(s.mean_orbit_altitude_m for s in samples) / len(samples) if samples else 0,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
