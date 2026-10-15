from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline
from .simulation.config import SimulationConfig
from .simulation.experiment import run_simulation_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Starlink TLE trajectory dataset.")
    subparsers = parser.add_subparsers(dest="command")

    data = subparsers.add_parser("data", help="Build the Starlink TLE trajectory dataset.")
    data.add_argument("--repo-root", type=Path, default=Path.cwd())
    data.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    data.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    data.add_argument("--reports-dir", type=Path, default=Path("reports"))
    data.add_argument("--source-tle", type=Path)
    data.add_argument("--no-download", action="store_true")
    data.add_argument("--allow-current-epochs", action="store_true")
    data.add_argument("--target-samples", type=int, default=1200)
    data.add_argument("--target-concurrent-samples", type=int, default=300)
    data.add_argument("--train-ratio", type=float, default=0.8)
    data.add_argument("--min-orbit-altitude-km", type=float, default=520.0)
    data.add_argument("--max-orbit-altitude-km", type=float, default=580.0)
    data.add_argument("--min-inclination-deg", type=float, default=52.5)
    data.add_argument("--max-inclination-deg", type=float, default=53.5)
    data.add_argument("--site-lat", type=float, default=0.0)
    data.add_argument("--site-lon", type=float, default=0.0)
    data.add_argument("--site-alt", type=float, default=0.0)

    sim = subparsers.add_parser("simulate", help="Run the full paper simulation suite.")
    sim.add_argument("--repo-root", type=Path, default=Path.cwd())
    sim.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    sim.add_argument("--reports-dir", type=Path, default=Path("reports"))
    sim.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    sim.add_argument("--trials", type=int, default=20)
    sim.add_argument("--num-uavs", type=int, default=20)
    sim.add_argument("--duration", type=float, default=60.0)
    sim.add_argument("--seed", type=int, default=202604)
    sim.add_argument("--realtime-only", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "simulate":
        config = SimulationConfig(
            repo_root=args.repo_root,
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            figures_dir=args.figures_dir,
            trials=args.trials,
            num_uavs=args.num_uavs,
            simulation_duration_s=args.duration,
            random_seed=args.seed,
        )
        report = run_simulation_suite(config)
    else:
        config = PipelineConfig(
            repo_root=args.repo_root,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            source_tle_path=args.source_tle,
            download_if_missing=not args.no_download,
            strict_year_filter=not args.allow_current_epochs,
            min_orbit_altitude_km=args.min_orbit_altitude_km,
            max_orbit_altitude_km=args.max_orbit_altitude_km,
            min_inclination_deg=args.min_inclination_deg,
            max_inclination_deg=args.max_inclination_deg,
            target_samples=args.target_samples,
            target_concurrent_samples=args.target_concurrent_samples,
            train_ratio=args.train_ratio,
            default_site_lat_deg=args.site_lat,
            default_site_lon_deg=args.site_lon,
            default_site_alt_m=args.site_alt,
        )
        report = run_pipeline(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
