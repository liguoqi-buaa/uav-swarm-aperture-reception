# UAV-swarm distributed-aperture reception of LEO satellites — simulation code and data
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23400496.svg)](https://doi.org/10.5281/zenodo.23400496)

Code, configuration, orbital data and analysis scripts for the paper

> **Predictive UAV-swarm aperture reception of LEO satellites: pitfalls, trade-offs, and a simple baseline**
> *(under review)*

We study ground-based **distributed-aperture reception** of non-cooperative LEO
satellites by a UAV swarm: the swarm forms a linear receiving array above a fixed
site and reorients to stay normal to the incoming wave direction. The task matters
because the formulation usually adopted in the literature — steering the swarm to
track the sub-satellite point — is physically infeasible: at a 550 km pass the
sub-satellite point moves at ~6.7 km/s, about **336 times** the speed of a
practical multirotor.

## Main findings (all reproducible from this repository)

1. **The alignment metric is confounded by array length.** A measured improvement of
   114.9% over the best same-law profile falls to 75.7% when all swarms are rescaled
   to a common baseline, and to 55.1% for the orientation-only component. Length-controlled,
   the predictive controller is *worse* than a simple geometric controller.
2. **The reference is chosen by the authors.** The rate-limited reference departs from the
   instantaneous optimum by more than 30° for 33% of the window, so "near-perfect alignment"
   can mean faithful tracking of a suboptimal target.
3. **Prediction is not the binding constraint.** Two prediction-free, learning-free
   geometric controllers reach an aperture gain of 0.569 against 0.281 for a
   physics-informed predictive controller, at **119× lower computation**. Adding a properly
   constrained receding-horizon preview *reduces* the gain by 4.2% (*p* < 0.001).
4. **The conclusions are site-independent and fit a real-time budget.** Reproduced at three
   ground sites from the same TLE archive; every controller fits a 1 s control period, and
   even the predictive controller would survive a conservative tenfold per-core slowdown.

## Repository layout

```
uav_swarm_scenario_C/     simulation project (the closed-loop simulator)
├── tle_swarm_sim/        TLE pipeline (SGP4), simulation core, controllers, metrics
├── src/                  ST-GNN model, feature builder, DMPC controller
├── configs/  tests/  tools/
├── data/                 orbital data actually used (see DATA_PROVENANCE.md)
├── models/stgnn/         trained ST-GNN checkpoints
├── scripts/              smoke test and reference training entry point
└── reports/              aggregate result tables (no raw trajectories)
analysis/                 scripts that regenerate every table and figure of the paper
├── *.py                  run / analyse / plot
├── covisible_pool.json   reference-site covisible window pool
└── pools/                window pools for the two additional ground sites
figures/                  paper figures and the graphical abstract
docs/
├── REPRODUCING.md        table-by-table, figure-by-figure reproduction commands
└── DATA_PROVENANCE.md    exact provenance of the orbital data
```

## Requirements

Python ≥ 3.10. The results in the paper were produced with Python 3.13.14 and
`numpy 2.5.2, pandas 3.0.1, scipy 1.18.1, networkx 3.7, sgp4 2.27, osqp 1.1.3,
torch 2.14.1+cpu, matplotlib 3.11.1` (see `requirements.txt`).

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e uav_swarm_scenario_C              # installs the `tle_swarm_sim` package
```

**No GPU is required.** Every experiment runs on CPU; the per-step cost of the
heaviest controller is 30.6 ms on a single desktop core.

## Quick start

Regenerate the main gain-matched comparison (5 profiles × 20 paired trials,
600 s closed loop each; ≈ 2 minutes on 10 CPU workers):

```bash
python analysis/run_scenario_c.py --mode matched --trials 20 --workers 10
python analysis/normalize_alignment.py --mode matched
```

The second command re-derives the length-controlled and orientation-only
alignment qualities from the stored element trajectories. Use `--min-elev 10`
(the paper's weight reference) if you call the module directly; the CLI default
of the standalone tool differs.

See **`docs/REPRODUCING.md`** for the complete mapping from every table and
figure in the paper to its command.

## Data

Orbital input is a **single-snapshot public CelesTrak Starlink GP/TLE query**
(`GROUP=starlink&FORMAT=tle`): 10,304 records covering 10,304 distinct satellites,
with epochs spanning 2026-04-29 to 2026-05-02 (a snapshot, *not* a time series).
The simulator uses a filtered subset (`data/processed/starlink_archive.tle`,
1,329 records: inclination 52.5–53.5°, mean altitude 520–580 km). Details and
attribution are in **`docs/DATA_PROVENANCE.md`**. TLE data are © CelesTrak
(T.S. Kelso) and are redistributed here under their terms of use.

## What is *not* in this repository

Per-trial element trajectories and time series (≈ 1 GB of CSV) are omitted to keep
the repository light. They are **not needed** to reproduce the results — re-running
the two quick-start commands regenerates them — and are available on request or
from the archived dataset release (Zenodo DOI once assigned).

## License

- **Code** (`uav_swarm_scenario_C/`, `analysis/`): MIT — see `LICENSE`.
- **Data and results** (`data/`, `figures/`, aggregate CSVs): CC BY 4.0 — see `DATA_LICENSE.md`.
  The upstream TLE data remain subject to CelesTrak's terms.

## Citation

See `CITATION.cff`. Once the paper is accepted, please cite the journal article;
until then please cite this repository and the archived release DOI.

## Contact

Corresponding authors: Guoqi Li (gqli@buaa.edu.cn), Sai Huang (huangsai@bupt.edu.cn).
