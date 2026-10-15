# Reproducing the paper

All commands are run **from the repository root**. Times are for a single
commodity CPU (16 logical cores); no GPU is used anywhere.

## 0. Setup

```bash
pip install -r requirements.txt
pip install -e uav_swarm_scenario_C
python -c "import sys; sys.path.insert(0, 'uav_swarm_scenario_C'); import tle_swarm_sim; print('ok')"
```

Optional sanity check of the TLE pipeline (a few seconds):

```bash
python uav_swarm_scenario_C/scripts/run_closed_loop_smoke.py
```

## 1. Mapping: paper item -> command

| Paper item | Command | Runtime |
|---|---|---|
| Table 1 (control profiles) | values are configuration scalars in `tle_swarm_sim/simulation/controllers.py` | — |
| Table 2 (compute budget) | `python analysis/profile_compute_budget.py --trials 3` | ~1.5 min |
| Table 3 (λ_min grid) | `python analysis/derive_lambda_min.py` | seconds |
| Table 4 (simulation parameters) | `uav_swarm_scenario_C/configs/paper_table1_config.yaml` + `tle_swarm_sim/simulation/config.py` | — |
| Table 5 (connectivity–aperture Pareto) | `python analysis/run_o1_blend.py` | ~5 min |
| Table 6 (three controller families) | `python analysis/run_scenario_c.py --mode matched --trials 20 --workers 10` then `python analysis/run_independent_baselines.py --mode matched --trials 20 --workers 10 --export-trials` | ~4 min |
| Table 7 (cross-site robustness) | see step 3 below | ~6 min |
| Fig. 1 (length confound) | `python analysis/normalize_alignment.py --mode matched`, then `python analysis/make_figures_cep.py` | ~1 min |
| Fig. 2 (reference suboptimality) | `python analysis/make_figures_cep.py` | included above |
| Fig. 3 (Pareto front) | `python analysis/make_figures_cep.py` | included above |
| Fig. 4 (three families) | `python analysis/make_figures_cep.py` | included above |
| Graphical abstract | `python analysis/make_graphical_abstract.py` | seconds |

`make_figures_cep.py` writes vector PDFs into `figures/`.

## 2. Length-controlled alignment (the 114.9% -> 75.7% result)

`analysis/normalize_alignment.py` re-derives, from the stored element
trajectories, three alignment qualities: the measured one, one at a common array
length, and one for orientation alone.

```bash
python analysis/run_scenario_c.py --mode matched --trials 20 --workers 10
python analysis/normalize_alignment.py --mode matched --root uav_swarm_scenario_C
```

The weight reference elevation is **10°** (equal to
`aperture_min_elevation_deg` in the configuration). If you call
`per_trial_metrics()` directly, set `normalize_alignment.MIN_ELEV = 10.0`
first; the standalone default is 15° and shifts the derived quantities by ~2%.

## 3. Cross-site robustness (Table 7)

The three ground sites use the **same TLE archive**; only the site coordinates,
and hence the covisible-window pool, change.

```bash
# 3a. window pools for the two additional sites (about 80 s each, single process)
python analysis/build_pool_sites.py --sites "mid30_west:30.0:-100.0" "mid45_europe:45.0:8.0"

# 3b. closed-loop sweeps: 9 controllers x 20 trials per site (about 2 min each)
python analysis/run_multisite.py --site-name mid30_west  --lat 30.0 --lon -100.0 \
    --pool analysis/pools/covisible_pool_mid30_west.json  --trials 20 --workers 10
python analysis/run_multisite.py --site-name mid45_europe --lat 45.0 --lon 8.0 \
    --pool analysis/pools/covisible_pool_mid45_europe.json --trials 20 --workers 10

# 3c. cross-site analysis -> analysis/multisite_summary.csv
python analysis/analyze_multisite.py
```

Note that `scenario_c.aperture_geometry()` reads the site coordinates from the
configuration, so `--lat/--lon` must match the pool file.

The reference-site results are read from the runs produced in steps 1–2.

## 4. What each script needs

Two kinds of scripts are shipped, and they have different inputs:

| Needs only the shipped data | Needs per-trial trajectories |
|---|---|
| `build_pool_sites.py`, `run_scenario_c.py`, `run_independent_baselines.py`, `run_multisite.py`, `run_o1_blend.py`, `train_stgnn_scenario_c.py`, `profile_compute_budget.py` | `normalize_alignment.py`, `analyze_multisite.py`, `analyze_independent.py`, `derive_lambda_min.py`, `compare_retrained.py` |

The second group reads `.../trials/<run>/<controller>/<trial>/trajectory_*.csv`.
Those files (~1 GB in total) are **not** shipped here; regenerate them by running
the corresponding `run_*.py` command in steps 1–3. The **aggregate** per-trial
files that carry the paper's reported metrics *are* shipped, under
`uav_swarm_scenario_C/reports/<run>/raw_trials/scenario_c_trials.csv`, together
with the derived CSVs in `analysis/` (`multisite_summary.csv`,
`alignment_normalized_*.csv`, `combined_comparison.csv`) — so every number in the
paper can be checked without re-running anything.

## 5. Reproducibility contract

* Every stochastic element (initial deployment, measurement noise, predicted-gradient
  noise, window selection) is driven by a deterministic seed schedule indexed by the
  trial number, so each trial is exactly reproducible.
* The same trial index selects the same window for every controller, so comparisons
  are paired by construction.
* All reported quantities are computed from the raw per-trial files, not from
  post-processed summaries.
* Determinism was verified across two independent Python environments; the
  length-controlled re-analysis reproduces the reported values to machine precision.
