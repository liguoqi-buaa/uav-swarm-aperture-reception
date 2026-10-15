from __future__ import annotations

import json

from .config import PipelineConfig
from .dataset import augment_concurrency, build_pass_samples, split_dataset, write_dataset, write_summary
from .downloader import download_starlink_tle
from .sgp4_utils import mean_orbit_altitude_m
from .tle import load_tle_records, write_tle_archive


def run_pipeline(config: PipelineConfig | None = None) -> dict:
    config = (config or PipelineConfig()).resolve()
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    source_path = config.source_tle_path or (config.raw_dir / f"{config.tle_group}.tle")
    if not source_path.exists():
        if not config.download_if_missing:
            raise RuntimeError(
                f"TLE source file does not exist: {source_path}. "
                "Place a CelesTrak Starlink TLE archive there or enable download."
            )
        try:
            download_starlink_tle(config.tle_url, source_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download CelesTrak Starlink TLE from {config.tle_url}. "
                f"Place a CelesTrak Starlink TLE snapshot at {source_path} and rerun."
            ) from exc

    all_records = load_tle_records(source_path)
    year_filtered_records = [
        record
        for record in all_records
        if not config.strict_year_filter or config.start_year <= record.epoch_year <= config.end_year
    ]
    records = [
        record
        for record in year_filtered_records
        if config.min_inclination_deg <= record.inclination_deg <= config.max_inclination_deg
        and config.min_orbit_altitude_km <= mean_orbit_altitude_m(record) / 1000.0 <= config.max_orbit_altitude_km
    ]
    if not records:
        raise RuntimeError(
            f"No usable Starlink TLE records parsed from {source_path}. "
            f"Expected epochs in {config.start_year}-{config.end_year} and "
            f"main-shell altitude/inclination filters."
        )

    write_tle_archive(records, config.processed_dir / "starlink_archive.tle")
    samples = build_pass_samples(records, config)
    concurrent_target = min(config.target_concurrent_samples, config.target_samples // 4)
    single_target = config.target_samples - concurrent_target
    concurrent_samples = augment_concurrency(samples, config)[:concurrent_target]
    if len(samples) < single_target:
        raise RuntimeError(
            f"Insufficient visible pass samples: got {len(samples)}, need at least {single_target}"
        )
    if len(concurrent_samples) < concurrent_target:
        raise RuntimeError(
            f"Insufficient multi-satellite concurrent samples: got {len(concurrent_samples)}, "
            f"need at least {concurrent_target}. Add denser CelesTrak historical TLE snapshots."
        )
    samples = samples[:single_target] + concurrent_samples
    train_samples, test_samples = split_dataset(samples, config.train_ratio)

    write_dataset(samples, config.processed_dir / "all_samples.json")
    write_dataset(train_samples, config.processed_dir / "train_samples.json")
    write_dataset(test_samples, config.processed_dir / "test_samples.json")
    write_summary(samples, config.reports_dir / "dataset_summary.json")

    report = {
        "source_tle": str(source_path),
        "source_records": len(all_records),
        "usable_epoch_records": len(year_filtered_records),
        "usable_main_shell_records": len(records),
        "archive_tle": str(config.processed_dir / "starlink_archive.tle"),
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "report": str(config.reports_dir / "dataset_summary.json"),
    }
    (config.reports_dir / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
