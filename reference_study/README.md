# Reference construction study — code, data and figures

This directory holds everything behind the study **"Reference construction biases
comparative simulation of UAV-swarm controllers: a proven failure mode, an exact
remedy, and a verified re-analysis"**: the reference generator, the three
experiments, every aggregate result table, and the figure script.

It is an **increment** on top of the closed-loop simulator in
[`../uav_swarm_scenario_C/`](../uav_swarm_scenario_C/) and the window pools in
[`../analysis/`](../analysis/). Nothing in those directories is modified.

---

## 1. What the study adds

Comparative simulation scores controllers against a reference that the simulator
itself produces. This work shows that the reference in use — a rate-limited greedy
follower that turns toward the instantaneous optimum at the maximum rotation rate —
**has a failure mode**: whenever the optimum migrates faster than that rate, the
follower's lag grows linearly and without bound, so the reference stops being
near-optimal.

The remedy is a **rate-constrained optimal reference**, obtained by backward dynamic
programming on a discretised orientation circle (period π, since a linear array is
invariant under a 180° flip). The consequence is not a uniform improvement: because
the aperture gain is *alignment × coverage* measured along the reference, correcting
the reference raises the reported gain of geometric controllers and lowers that of
predictive and parametric ones — which **strengthens** rather than undermines the
original conclusion.

---

## 2. Layout

```
reference_study/
├── code/
│   ├── common.py                  site definitions, SimulationConfig, geometry cache loader
│   ├── refgenerator.py            ★ the generator: rate-constrained DP on the π-circle
│   ├── exp1_ref_quality.py        experiment 1 — reference quality + correctness checks
│   ├── exp2_metric_recompute.py   experiment 2 — re-analysis under a corrected reference
│   ├── exp3_planner.py            experiment 3 — planning vs prediction (inconclusive)
│   ├── make_figures_simpat.py     regenerates all three figures
│   └── verify_numbers.py          re-derives every number quoted in the manuscript
├── experiments/                   all aggregate results (CSV/JSON/NPZ, ~2.6 MB)
└── figures/                       Figures 1–3 (vector PDF, single-column width)
```

`common.py` detects the layout automatically: it looks for the window pools in
`../analysis/` (this repository) or `_c_work/` (the authors' working tree) and for
results in `experiments/` or `02_experiments/`, so the same scripts run in both
without editing a path.

**Language.** Every module docstring — purpose, method, arguments, outputs — is in
English, as are all identifiers. Some *inline* comments are in Chinese (the authors'
development language); they annotate implementation detail only, never a definition or
a result.

---

## 3. Which script produces which table or figure

| Manuscript item | Script | Command |
|:--|:--|:--|
| **Table 1** — reference quality in use per site (`at rate limit`, `q`, `q of optimum`, `Δθ > 30°`, `collapsed`) | `exp1_ref_quality.py` | `--trials 20 --grid 3600` |
| **Table 2** — grid convergence (900 / 1800 / 3600 / 7200) | `exp1_ref_quality.py` | `--trials 20 --grid-check` |
| **Table 4** — aperture gain under `v_ref` vs `v_opt`, nine controllers × three sites | `exp2_metric_recompute.py` | `--trials 20 --grid 3600` |
| §5.4 correctness checks (four independent checks) | `exp1_ref_quality.py` | printed + `exp1_ref_quality.csv` |
| §5.5 the two gain aggregations | `exp1_ref_quality.py` | `gain_pct_mean`, `gain_pct_aggregate` |
| §6.3 projection spans per controller family | `exp2_metric_recompute.py` | `span_ref_m` / `span_opt_m` columns |
| §7 planning versus prediction (inconclusive) | `exp3_planner.py` | `--trials 20 --grid 3600 --site <name>` |
| **Figures 1–3** | `make_figures_simpat.py` | no arguments |

**Two aggregations, both stored.** §5.5 of the manuscript reports the reference-quality
gain two ways, and they answer different questions:

| Column | Equator / 30°N / 45°N | Manuscript wording |
|:--|:--|:--|
| `gain_pct_mean` — mean of per-window ratios | 61.3 / 26.2 / 16.5 % | *"improves by 16 % to 61 % depending on site"* |
| `gain_pct_aggregate` — ratio of summed totals | 31.3 / 21.3 / 14.9 % | *"the ratio of summed quality (+31.3 %, +21.3 %, +14.9 %)"* |

Table 2 (grid convergence) quotes the **aggregate** form (26.64 / 30.29 / 31.27 / 31.27).
Table 1 contains **no gain column** — it reports `q` at the reference, `q` at the optimum,
the saturated-step fraction, the Δθ > 30° fraction and the collapsed-window count.

> `exp1_grid_convergence.csv` also carries `seconds` and `seconds_per_window`. Those are
> **wall-clock on the archiving machine** and vary by a factor of 3–3.5 between runs; do
> not treat them as a result. Every other column reproduces bit-for-bit.

---

## 4. How to run

### 4.1 Experiment 1 and the figures — self-contained

These need only what is already in this repository:

```bash
cd reference_study/code
python exp1_ref_quality.py --trials 20 --grid 3600        # ≈ 36 s on one core
python make_figures_simpat.py                             # < 10 s
```

`exp1` reads the cached geometry in `experiments/geometry_*.npz`; if a cache is
missing it rebuilds it from the TLE data via SGP4 (slower, same result).

### 4.2 Experiments 2 and 3 — need the closed-loop run first

Both are **re-analyses** of element trajectories produced by the earlier closed-loop
study. Those per-step trajectories are deliberately **not** shipped here (≈1 GB);
only the aggregate tables are. Regenerate them first, from the repository root:

```bash
python analysis/run_scenario_c.py --mode matched     --trials 20 --workers 10
python analysis/run_scenario_c.py --mode independent --trials 20 --workers 10
```

Then:

```bash
cd reference_study/code
python exp2_metric_recompute.py --trials 20 --grid 3600
python exp3_planner.py --trials 20 --grid 3600 --site eq_pacific
python exp3_planner.py --trials 20 --grid 3600 --site mid30_west
python exp3_planner.py --trials 20 --grid 3600 --site mid45_europe
```

If the prerequisite directories are absent, each script stops with a message naming
the missing path and the command that produces it, rather than failing mid-way.

> Note on experiment 3: its outcome is **inconclusive** — the sign of the effect
> flips across sites (+10.9% / −0.0% / −9.5%) and a residual 1.5% deviation in the
> late closed-loop segment was not traced. It is reported as such in the manuscript
> and is **not** part of the main argument. Do not read it as a validation.

---

## 5. What was verified

The repository is meant to be checked, not trusted. Two verifications were run
before archiving:

| Check | Result |
|:--|:--|
| `exp1` re-run from the cached geometry, all 60 windows at grid 3600 | **bit-identical** to the archived `exp1_ref_quality.csv` and `exp1_scenarios.csv` — max absolute deviation **0.000e+00** over 39 numeric columns (timing columns excluded) |
| Figures regenerated from the archived results, rendered at 150 dpi | **pixel-identical** to the published PDFs (max channel difference 0) |
| `q_ref` reconstructed here vs. the simulator's own `oracle_gain_step` | max deviation **1.5e-14** |
| Single-step turn of every recovered trajectory vs. the rate limit | 0.750° ≤ 0.7639° at every window and every grid |
| `exp2` re-analysis vs. the stored per-step records (in the reference in use) | coverage **2.4e-14**, alignment **4.4e-16**, gain **2.1e-14** |

`verify_numbers.py` re-derives every quantity quoted in the manuscript and compares
it against the `.tex` source. It is included so the check can be repeated; it needs
the manuscript, which is **not** part of this repository:

```bash
# put manuscript.tex somewhere, then
VERIFY_TEX=/path/to/manuscript.tex python verify_numbers.py
```

**Timing figures are environment-dependent.** Re-running the same code on the same
machine gave per-window DP times differing by a factor of 3–3.5 (CPU load and thermal
state). The manuscript therefore quotes a conservative upper bound ("under two
seconds per window on one core") rather than exact seconds. Every scientific value
in it, by contrast, reproduces bit-for-bit.

---

## 6. Data

| File | Content |
|:--|:--|
| `geometry_{site}_n20.npz` | per-window geometry cache: azimuths `phi`, priorities `w`, validity `valid`, reference and optimum orientations `th_ref`/`th_dem`, reference quality `q_ref`, and the simulator's own `oracle` |
| `vopt_{site}_n20_g3600.npz` | optimal reference trajectories at grid 3600 |
| `exp1_ref_quality.csv` | one row per window: `q_ref`, `q_opt`, gain, saturated-step fraction, gap-to-optimum fraction, the four correctness checks, and the DP wall time |
| `exp1_scenarios.csv` | per-site aggregation, both gain aggregations |
| `exp1_grid_convergence.csv` | grid 900 / 1800 / 3600 / 7200 |
| `exp2_metric_recompute.csv` | per-step recomputed metrics for every controller, site and trial |
| `exp2_by_controller.csv` / `exp2_ranking.csv` | per-controller means under both references, and the ranking |
| `exp3_planner_{site}.csv`, `exp3_by_controller_{site}.csv` | experiment 3 results, **one file per site** |

Output naming note: every per-site experiment writes the site name into the filename.
An earlier version hard-coded it and silently let three sites overwrite one another;
that bug is why the naming is enforced here.

---

## 7. Environment

Python ≥ 3.10 with `numpy`, `pandas`, `scipy`, `matplotlib` (plus the simulator's own
dependencies, see [`../requirements.txt`](../requirements.txt)). The archived results
were produced with Python 3.13.14, NumPy 2.5.2, pandas 3.0.1, matplotlib 3.11.1.
**No GPU is required**; every experiment in this directory runs on a single CPU core.

---

## 8. Licence

Code MIT, results CC BY 4.0 — see [`../LICENSE`](../LICENSE) and
[`../DATA_LICENSE.md`](../DATA_LICENSE.md).
