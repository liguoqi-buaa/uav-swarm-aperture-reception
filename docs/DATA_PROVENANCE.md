# Orbital data provenance

## The file used by the simulator

| File | Records | Contents |
|---|---:|---|
| `uav_swarm_scenario_C/data/raw/starlink.tle` | 10,304 | one TLE per satellite, 10,304 distinct `catalog_number`s |
| `uav_swarm_scenario_C/data/processed/starlink_archive.tle` | 1,329 | main-shell subset used by the simulator |

## Source

* **Provider:** CelesTrak, `https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle`
* **Acquisition:** a single snapshot query. It is **not** a historical archive and
  **not** a time series — each satellite appears exactly once.
* **Epoch range:** **2026-04-29 to 2026-05-02** (about three days; the spread
  reflects each satellite's most recent element set, not a time series).
* **Derived archive:** the simulator applies main-shell filters
  (inclination 52.5–53.5°, mean altitude 520–580 km), which yields 1,329 records
  with epochs 2026-04-30 to 2026-05-01.

> Please do **not** describe this dataset as a "2024–2025 historical TLE archive".
> It is a 2026 snapshot; the epoch range above is the correct statement.

## Covisible window pools

`analysis/covisible_pool.json` (reference site, 0.0°N 174.624885°W) and
`analysis/pools/covisible_pool_*.json` (30.0°N 100.0°W and 45.0°N 8.0°E) each
contain 40 covisible three-satellite windows of 600 s, selected from the
candidate windows of the same archive at the corresponding site. Each window
records the satellite catalogue numbers, the valid fraction above the 10°
elevation floor, and the minimum sub-satellite-point distance (used to exclude
near-zenith geometry, where the azimuth is ill-conditioned).

Regenerate them with:

```bash
python analysis/build_pool_sites.py --sites "eq_pacific:0.0:-174.624885" \
                                           "mid30_west:30.0:-100.0" \
                                           "mid45_europe:45.0:8.0"
```

## Reproduction of the dataset from source

```bash
python -m tle_swarm_sim.cli data --source-tle uav_swarm_scenario_C/data/raw/starlink.tle \
    --no-download --allow-current-epochs
```

`--allow-current-epochs` is required: the template year filter in
`tle_swarm_sim/config.py` defaults to 2024–2025 and would otherwise select an
empty set for this 2026 snapshot.

## Attribution

TLE/GP data are published by CelesTrak and should be acknowledged as
(T.S. Kelso, CelesTrak, https://celestrak.org). Orbit propagation uses the `sgp4`
library (Vallado et al., 2006).
