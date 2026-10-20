# UAV-swarm distributed-aperture reception of LEO satellites — simulation code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23400496.svg)](https://doi.org/10.5281/zenodo.23400496)

Code, orbital data, experiment results and figure scripts for a study of
**ground-based distributed-aperture reception of non-cooperative LEO satellites by a
UAV swarm**: the swarm forms a linear receiving array above a fixed site and
reorients to stay normal to the incoming wave direction.

The task matters because the formulation usually adopted in the literature — steering
the swarm to track the sub-satellite point — is physically infeasible: at a 550 km pass
the sub-satellite point moves at ~6.7 km/s, about **336 times** the speed of a practical
multirotor. What the swarm can do is reorient quickly, which makes the *reference
direction* it is commanded to hold the quantity that governs everything downstream.

---

## Two stages, two manuscripts

This repository accumulates two successive studies on the same simulator. Both are
reproducible from what is here.

### Stage 2 (current) — `reference_study/`

> **Reference construction biases comparative simulation of UAV-swarm controllers:
> a proven failure mode, an exact remedy, and a verified re-analysis**

Comparative simulation scores controllers against a reference that the simulator itself
produces. That reference had never been examined. This stage shows it is the weak link:

1. **The reference in use has a failure mode.** It is a rate-limited greedy follower that
   turns toward the instantaneous optimum at the maximum rotation rate. Whenever the
   optimum migrates faster than that rate, the lag grows linearly and without bound, so
   the reference stops being near-optimal — with a closed form for the time to reach the
   antipode, `T* = (π − ℓ₀)/(ν − ω_max)`. Measured: the optimum reaches
   **89.5°/s** against a limit of **0.764°/s** (117×), and the construction sits at the
   limit for **74–83 %** of steps and collapses in **20 %** of equatorial windows.
2. **An exact remedy.** Reference generation is posed as a rate-constrained optimal
   control problem on the periodic orientation circle and solved exactly by backward
   dynamic programming — under two seconds per 600 s window on one CPU core.
3. **Correcting it changes what is measured — and strengthens the earlier conclusion.**
   Because the aperture gain is *alignment × coverage* measured along the reference,
   the correction moves geometric controllers up (−1 to +11 %) and predictive and
   parametric ones down (−4 to −15 %). The geometric-over-predictive advantage grows
   from **2.03× to 2.31×**, and the defective reference turns out to have been
   *flattering* the predictive controller (its misalignment under a reachable target
   rises from 11.4° to 28.4°). The ranking of nine controllers is unchanged across
   three ground sites.

### Stage 1 — `uav_swarm_scenario_C/` + `analysis/`

> **Predictive UAV-swarm aperture reception of LEO satellites: pitfalls, trade-offs,
> and a simple baseline** (earlier manuscript; its results remain valid and are
> re-analysed in stage 2)

1. **The alignment metric is confounded by array length.** A measured improvement of
   114.9 % over the best same-law profile falls to 75.7 % when all swarms are rescaled to
   a common baseline, and to 55.1 % for the orientation-only component.
2. **The reference is chosen by the authors.** The rate-limited reference departs from the
   instantaneous optimum by more than 30° for 33 % of the window, so "near-perfect
   alignment" can mean faithful tracking of a suboptimal target. *(Stage 2 turns this
   observation into the main subject.)*
3. **Prediction is not the binding constraint.** Two prediction-free, learning-free
   geometric controllers reach an aperture gain of 0.569 against 0.281 for a
   physics-informed predictive controller, at **119× lower computation**. Adding a
   properly constrained receding-horizon preview *reduces* the gain by 4.2 % (*p* < 0.001).
4. **The conclusions are site-independent and fit a real-time budget.** Reproduced at three
   ground sites from the same TLE archive; every controller fits a 1 s control period, and
   even the predictive controller would survive a conservative tenfold per-core slowdown.

---

## Repository layout

```
reference_study/          stage 2 — reference construction study
├── code/                 reference generator (DP on the π-circle), three experiments,
│                         figure script, number-verification script
├── experiments/          every aggregate result behind stage 2's tables and figures
├── figures/              stage 2 figures (vector PDF, single-column width)
└── README.md             ★ table-by-table reproduction guide for stage 2

uav_swarm_scenario_C/     simulation project (the closed-loop simulator)
├── tle_swarm_sim/        TLE pipeline (SGP4), simulation core, controllers, metrics
├── src/                  ST-GNN model, feature builder, DMPC controller
├── configs/  tests/  tools/
├── data/                 orbital data actually used (see docs/DATA_PROVENANCE.md)
├── models/stgnn/         trained ST-GNN checkpoints
├── scripts/              smoke test and reference training entry point
└── reports/              aggregate result tables (no raw trajectories)

analysis/                 stage 1 scripts that regenerate every table and figure
├── *.py                  run / analyse / plot
├── covisible_pool.json   reference-site covisible window pool
└── pools/                window pools for the two additional ground sites

figures/                  stage 1 figures and the graphical abstract
docs/
├── REPRODUCING.md        stage 1 reproduction commands, table by table
└── DATA_PROVENANCE.md    exact provenance of the orbital data
```

---

## Requirements

Python ≥ 3.10. The results were produced with Python 3.13.14 and
`numpy 2.5.2, pandas 3.0.1, scipy 1.18.1, networkx 3.7, sgp4 2.27, osqp 1.1.3,
torch 2.14.1+cpu, matplotlib 3.11.1` (see `requirements.txt`).

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e uav_swarm_scenario_C              # installs the `tle_swarm_sim` package
```

**No GPU is required.** Every experiment runs on CPU; the per-step cost of the heaviest
controller is 30.6 ms on a single desktop core.

---

## Quick start

**Stage 2 — the self-contained part** (needs nothing beyond this repository, ≈ 40 s):

```bash
cd reference_study/code
python exp1_ref_quality.py --trials 20 --grid 3600    # reference quality, all 3 sites
python make_figures_simpat.py                         # Figures 1–3
```

**Stage 1 — the closed-loop comparison** (5 profiles × 20 paired trials, 600 s closed
loop each; ≈ 2 minutes on 10 CPU workers):

```bash
python analysis/run_scenario_c.py --mode matched --trials 20 --workers 10
python analysis/normalize_alignment.py --mode matched
```

Use `--min-elev 10` (the paper's weight reference) if you call the module directly; the
CLI default of the standalone tool differs. See **`docs/REPRODUCING.md`** for the
complete mapping from every stage 1 table and figure to its command, and
**`reference_study/README.md`** for stage 2.

---

## Data

Orbital input is a **single-snapshot public CelesTrak Starlink GP/TLE query**
(`GROUP=starlink&FORMAT=tle`): 10,304 records covering 10,304 distinct satellites, with
epochs spanning 2026-04-29 to 2026-05-02 (a snapshot, *not* a time series). The simulator
uses a filtered subset (`data/processed/starlink_archive.tle`, 1,329 records:
inclination 52.5–53.5°, mean altitude 520–580 km). Details and attribution are in
**`docs/DATA_PROVENANCE.md`**. TLE data are © CelesTrak (T. Kelso) and are redistributed
here under their terms of use.

## What is *not* in this repository

Per-trial element trajectories and per-step time series (≈ 1 GB of CSV) are omitted to
keep the repository light. They are **not** needed to reproduce stage 2's Tables 1–2 or
Figures 1–3, which ship with their inputs; stage 2's experiments 2–3 are re-analyses of
those trajectories and do require regenerating them first — the two commands are given in
`reference_study/README.md`.

## Verification, not assertion

Two independent checks were run before archiving: re-running stage 2's experiment 1 from
the cached geometry reproduced the archived result tables **bit-for-bit** (max absolute
deviation `0.000e+00` over 39 numeric columns), and regenerating the figures from those
tables gave **pixel-identical** output. `reference_study/code/verify_numbers.py`
re-derives every quantity quoted in the manuscript. See `reference_study/README.md` §5.

## License

- **Code** (`uav_swarm_scenario_C/`, `analysis/`, `reference_study/code/`): MIT — see `LICENSE`.
- **Data and results** (`data/`, `figures/`, `reference_study/experiments/`, aggregate CSVs):
  CC BY 4.0 — see `DATA_LICENSE.md`. The upstream TLE data remain subject to CelesTrak's terms.

## Citation

See `CITATION.cff`. Once the manuscript is published, please cite the journal article;
until then cite this repository and the archived release DOI.

## Contact

Corresponding authors: Guoqi Li (gqli@buaa.edu.cn), Sai Huang (huangsai@bupt.edu.cn).
