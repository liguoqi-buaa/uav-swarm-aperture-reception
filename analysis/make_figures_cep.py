"""CEP manuscript figures (matplotlib, grayscale-safe, single-column sized).

Produces four vector PDFs into CEP_LaTeX/figures/:

  figA_pareto.pdf           G - lambda_2 Pareto front (connectivity_blend sweep + independent controllers)
  figB_families.pdf         three controller families on G / length-controlled A / lambda_2
  figC_reference_gap.pdf    angle between rate-limited reference v_ref and instantaneous demand v_demand
  figD_normalization.pdf    measured A vs length-controlled A vs orientation-only A_dir

Design rules (carried over from the CJA figure pass):
  - grayscale-progressive palette + non-colour encoding fallback (hatch / marker / linestyle)
  - figsize width ~6.5 in == elsarticle 3p textwidth, so in-figure fonts print at full size
  - 1.2 pt lines, 7.5-8 pt labels

Usage:
    python make_figures_cep.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
YZH = HERE.parent
ROOT = YZH / "uav_swarm_scenario_C"
REPORTS = ROOT / "reports"
FIGS = YZH / "figures"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- style ----
# Grayscale-progressive palette. The key baseline (B2) and the predictive
# controller (ST) sit at the two extremes of the luminance range.
COLORS = {
    "ST-GNN-DMPC":          "#8B1A1A",  # dark red     (lum  59.8)
    "B2_GREEDY":            "#1F4E79",  # dark blue    (lum  68.8)
    "B2_GREEDY_PREV__w2":   "#2E7D32",  # dark green   (lum  92.8)
    "B1_GEOM_PD":           "#B45309",  # dark orange  (lum 103.6)
    "P3":                   "#7E57C2",  # purple       (lum 110.9)
}
HATCHES = {
    "ST-GNN-DMPC":          "//",
    "B2_GREEDY":            "\\\\",
    "B2_GREEDY_PREV__w2":   "oo",
    "B1_GEOM_PD":           "..",
    "P3":                   "xx",
}
MARKERS = {
    "ST-GNN-DMPC":          "o",
    "B2_GREEDY":            "s",
    "B2_GREEDY_PREV__w2":   "^",
    "B1_GEOM_PD":           "D",
    "P3":                   "v",
}
LINESTYLES = {
    "ST-GNN-DMPC":          "-",
    "B2_GREEDY":            "--",
    "B2_GREEDY_PREV__w2":   "-.",
    "B1_GEOM_PD":           ":",
    "P3":                   (0, (3, 1, 1, 1)),
}
# display names (mathtext for subscripts)
DISPLAY = {
    "ST-GNN-DMPC":          "ST-GNN-DMPC",
    "B2_GREEDY":            "$B_2$",
    "B2_GREEDY_PREV__w2":   "$B_2^{p}$",
    "B1_GEOM_PD":           "$B_1$",
    "P3":                   "$P_3$",
}

FS_LABEL = 7.5
FS_TICK = 7.0
FS_LEGEND = 6.6
FS_TITLE = 7.8
FS_PANEL = 7.5

# ------------------------------------------------------------- raw data ----
TRIAL_FILES = {
    "P3":                  REPORTS / "scenario_c_matched/raw_trials/scenario_c_trials.csv",
    "B1_GEOM_PD":          REPORTS / "scenario_c_independent_matched_full/raw_trials/scenario_c_trials.csv",
    "B2_GREEDY":           REPORTS / "scenario_c_independent_matched_full/raw_trials/scenario_c_trials.csv",
    "B2_GREEDY_PREV__w2":  REPORTS / "scenario_c_independent_matched_prev/raw_trials/scenario_c_trials.csv",
    "ST-GNN-DMPC":         REPORTS / "scenario_c_matched/raw_trials/scenario_c_trials.csv",
}
RAW_NAME = {  # name used inside the raw csv
    "P3": "DMPC-C", "B1_GEOM_PD": "B1_GEOM_PD", "B2_GREEDY": "B2_GREEDY",
    "B2_GREEDY_PREV__w2": "B2_GREEDY_PREV__w2", "ST-GNN-DMPC": "ST-GNN-DMPC",
}
NORM_FILES = {
    "P3":                  HERE / "alignment_normalized_matched.csv",
    "B1_GEOM_PD":          HERE / "alignment_normalized_scenario_c_independent_matched_full.csv",
    "B2_GREEDY":           HERE / "alignment_normalized_scenario_c_independent_matched_full.csv",
    "B2_GREEDY_PREV__w2":  HERE / "alignment_normalized_scenario_c_independent_matched_prev.csv",
    "ST-GNN-DMPC":         HERE / "alignment_normalized_scenario_c_o1_blend0.65_matched.csv",
}
NORM_NAME = {"P3": "DMPC-C", "ST-GNN-DMPC": "ST-GNN-DMPC"}  # default = key


def raw_stats(algo: str) -> dict[str, float]:
    """Mean of selected metrics for one controller from its raw trial file."""
    f = TRIAL_FILES[algo]
    df = pd.read_csv(f)
    sub = df[df["algorithm"] == RAW_NAME[algo]]
    return {
        "n": len(sub),
        "G": sub["aperture_gain"].mean(),
        "G_sd": sub["aperture_gain"].std(ddof=1),
        "A": sub["aperture_alignment"].mean(),
        "C": sub["aperture_coverage"].mean(),
        "span": sub["aperture_span_m"].mean(),
        "mis": sub["aperture_misalignment_deg"].mean(),
        "lam2": sub["average_connectivity"].mean(),
        "lam2_sd": sub["average_connectivity"].std(ddof=1),
        "link": sub["link_availability_ratio"].mean(),
        "ms": sub["average_single_step_ms"].mean(),
    }


def norm_stats(algo: str) -> dict[str, float]:
    """Measured A, length-controlled A@3000 and orientation-only Adir@3000."""
    f = NORM_FILES[algo]
    df = pd.read_csv(f)
    name = NORM_NAME.get(algo, algo)
    sub = df[df["algorithm"] == name]
    return {
        "A_meas": sub["A_csv"].mean(),
        "A_3000": sub["A@3000"].mean(),
        "A_3000_sd": sub["A@3000"].std(ddof=1),
        "Adir_3000": sub["Adir@3000"].mean(),
        "Adir_3000_sd": sub["Adir@3000"].std(ddof=1),
    }


def pareto_points() -> list[tuple[str, float, float, float]]:
    """(label, G, lambda_2, coverage) for the blend sweep and the two extremes."""
    pts = []
    for bl in ["0.0", "0.2", "0.4", "0.65"]:
        f = REPORTS / f"scenario_c_o1_blend{bl}_matched/raw_trials/scenario_c_trials.csv"
        df = pd.read_csv(f)
        sub = df[df["algorithm"] == "ST-GNN-DMPC"]
        pts.append((bl, sub["aperture_gain"].mean(),
                    sub["average_connectivity"].mean(),
                    sub["aperture_coverage"].mean()))
    s = raw_stats("B2_GREEDY")
    pts.append(("$B_2$", s["G"], s["lam2"], s["C"]))
    p = raw_stats("P3")
    pts.append(("$P_3$", p["G"], p["lam2"], p["C"]))
    return pts


# ---------------------------------------------------- reference gap data ----
def reference_gap(n_trials: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_s, per-trial angle matrix (n_trials, T), elevation matrix).

    The angle is between the rate-limited reference v_ref (geo['v_star']) and the
    instantaneous geometric demand v_demand, in degrees folded to [0, 90].
    """
    from tle_swarm_sim.simulation.config import SimulationConfig
    from tle_swarm_sim.simulation.experiment import _load_records
    from tle_swarm_sim.simulation.scenario_c import (
        select_aperture_window, group_records_for_window, aperture_geometry)

    cfg = SimulationConfig(
        repo_root=ROOT, reports_dir=REPORTS / "_tmp", figures_dir=REPORTS / "_tmp",
        trials=20, concurrent_satellites=(3,), simulation_duration_s=600.0, control_step_s=1.0,
        scenario_mode="aperture", site_lat_deg=0.0, site_lon_deg=-174.624885,
        aperture_pool_path=HERE / "covisible_pool.json",
        aperture_baseline_m=3000.0, aperture_sigma_m=500.0, aperture_min_elevation_deg=10.0,
        monitoring_kernel_radius_m=1000.0, deployment_mode="monitor_area_near",
        monitor_area_initial_spread_m=1500.0, max_communication_range_m=2500.0,
        task_window_mode="extended_pass_10deg",
    ).resolve()

    records = _load_records(cfg)
    angs, elevs, t_axis = [], [], None
    for trial in range(n_trials):
        w = select_aperture_window(records, cfg, trial)
        if w is None:
            continue
        g = group_records_for_window(records, w)
        geo = aperture_geometry(g, datetime.fromisoformat(w["start_utc"]), cfg)
        vd, vr = geo["v_demand"], geo["v_star"]
        cos = np.abs((vd * vr).sum(1))
        ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
        angs.append(ang)
        elevs.append(geo["elev_deg"])
        if t_axis is None:
            t_axis = np.arange(len(ang)) * cfg.control_step_s
    n = min(a.shape[0] for a in angs)
    n = min(n, *(e.shape[0] for e in elevs))
    return t_axis[:n], np.vstack([a[:n] for a in angs]), np.vstack([e[:n] for e in elevs])


def _panel(ax, letter: str) -> None:
    ax.text(0.015, 0.96, f"({letter})", transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", va="top", ha="left")


def _bar_style(algos):
    return ([COLORS[a] for a in algos], [HATCHES[a] for a in algos])


# ------------------------------------------------------------------ figs ----
def figA() -> None:
    pts = pareto_points()
    labels = [p[0] for p in pts]
    G = np.array([p[1] for p in pts])
    lam = np.array([p[2] for p in pts])
    cov = np.array([p[3] for p in pts])

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    # blend sweep (first four) sorted by lambda_2 for the front line
    sweep = pts[:4]
    sw_lam = np.array([p[2] for p in sweep]); sw_G = np.array([p[1] for p in sweep])
    idx = np.argsort(sw_lam)
    ax.plot(sw_lam[idx], sw_G[idx], color="#444444", lw=1.2, ls="--", marker="o",
            ms=5, mfc="white", mew=1.0, zorder=2,
            label="predictive family, $\\beta$ sweep")
    for lab, x, y in zip(labels[:4], sw_lam, sw_G):
        ax.annotate(f"$\\beta={lab}$", (x, y), textcoords="offset points",
                    xytext=(6, -10 if lab in ("0.0",) else 6), fontsize=FS_LEGEND)
    # independent baseline and profile
    ax.scatter([lam[4]], [G[4]], marker="s", s=55, c=COLORS["B2_GREEDY"],
               edgecolors="black", linewidths=0.6, zorder=3, label="$B_2$ (independent)")
    ax.scatter([lam[5]], [G[5]], marker="v", s=55, c=COLORS["P3"],
               edgecolors="black", linewidths=0.6, zorder=3, label="$P_3$ (profile)")
    ax.annotate("$B_2$", (lam[4], G[4]), textcoords="offset points", xytext=(8, -2),
                fontsize=FS_LEGEND + 0.5)
    ax.annotate("$P_3$", (lam[5], G[5]), textcoords="offset points", xytext=(8, -2),
                fontsize=FS_LEGEND + 0.5)
    ax.set_xlabel("Algebraic connectivity $\\lambda_2$", fontsize=FS_LABEL)
    ax.set_ylabel("Aperture gain $G$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="center right")
    _panel(ax, "a")
    fig.tight_layout()
    fig.savefig(FIGS / "figA_pareto.pdf")
    plt.close(fig)
    for lab, g, l, c in pts:
        print(f"  [A] {lab:<6} G={g:.4f}  lambda2={l:.2f}  C={c:.3f}")


def figB() -> None:
    algos = ["P3", "B1_GEOM_PD", "B2_GREEDY", "B2_GREEDY_PREV__w2", "ST-GNN-DMPC"]
    raw = {a: raw_stats(a) for a in algos}
    nrm = {a: norm_stats(a) for a in algos}

    specs = [
        ("G", [raw[a]["G"] for a in algos], [raw[a]["G_sd"] for a in algos],
         "Aperture gain $G$"),
        ("A3000", [nrm[a]["A_3000"] for a in algos], [nrm[a]["A_3000_sd"] for a in algos],
         "Length-controlled $A$ at $L=3000$ m"),
        ("lam2", [raw[a]["lam2"] for a in algos], [raw[a]["lam2_sd"] for a in algos],
         "Algebraic connectivity $\\lambda_2$"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 5.0))
    x = np.arange(len(algos))
    cols, hats = _bar_style(algos)
    for i, (ax, (_, mu, sd, ylab)) in enumerate(zip(axes, specs)):
        ax.bar(x, mu, yerr=sd, capsize=2.5, color=cols, edgecolor="black",
               linewidth=0.5, hatch=hats, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[a] for a in algos], fontsize=FS_TICK + 0.5)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.3, axis="y")
        _panel(ax, "abc"[i])
        if i < 2:
            ax.tick_params(labelbottom=False)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(FIGS / "figB_families.pdf")
    plt.close(fig)
    for a in algos:
        print(f"  [B] {DISPLAY[a]:<22} G={raw[a]['G']:.4f}  A@3000={nrm[a]['A_3000']:.4f}"
              f"  lam2={raw[a]['lam2']:.2f}  ms={raw[a]['ms']:.3f}")


def figC() -> None:
    t, ang, elev = reference_gap()
    mean = ang.mean(0)
    p95 = np.percentile(ang, 95, axis=0)
    allang = ang.reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.9))
    ax = axes[0]
    ax.fill_between(t, 0, p95, color="#1F4E79", alpha=0.22, label="95th percentile")
    ax.plot(t, mean, color="#1F4E79", lw=1.2, ls="-", marker="o", markevery=max(1, len(t) // 14),
            ms=3.0, mfc="white", mew=0.7, label="mean angle")
    ax.axhline(30.0, color="#8B1A1A", lw=1.2, ls="--", label="$30^\\circ$ threshold")
    ax.set_xlabel("Time (s)", fontsize=FS_LABEL)
    ax.set_ylabel("Angle (deg)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left", ncol=3)
    _panel(ax, "a")

    ax2 = axes[1]
    bins = np.linspace(0, 90, 19)
    ax2.hist(allang, bins=bins, color="#B45309", edgecolor="black", linewidth=0.5,
             hatch="..", alpha=0.9)
    ax2.axvline(30.0, color="#8B1A1A", lw=1.2, ls="--")
    frac = 100.0 * (allang > 30.0).mean()
    ax2.text(32, ax2.get_ylim()[1] * 0.85, f"{frac:.1f}% of samples $>30^\\circ$",
             fontsize=FS_LEGEND + 0.5, color="#8B1A1A")
    ax2.set_xlabel("Angle between $v_{\\mathrm{ref}}$ and $v_{\\mathrm{dem}}$ (deg)",
                   fontsize=FS_LABEL)
    ax2.set_ylabel("Count", fontsize=FS_LABEL)
    ax2.tick_params(labelsize=FS_TICK)
    ax2.grid(alpha=0.3, axis="y")
    _panel(ax2, "b")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(FIGS / "figC_reference_gap.pdf")
    plt.close(fig)
    print(f"  [C] mean={allang.mean():.1f}  median={np.median(allang):.1f}  "
          f"p95={np.percentile(allang, 95):.1f}  max={allang.max():.1f}  "
          f">30deg={frac:.1f}%  >10deg={100*(allang>10).mean():.1f}%  n={len(allang)}")
    pt = [100.0 * (row > 30.0).mean() for row in ang]
    pm = [row.mean() for row in ang]
    print(f"      per-trial mean angle: {min(pm):.1f}-{max(pm):.1f} (avg {np.mean(pm):.1f}); "
          f"per-trial >30deg: {min(pt):.1f}-{max(pt):.1f}% (avg {np.mean(pt):.1f}%)")


def figD() -> None:
    algos = ["P3", "B2_GREEDY", "ST-GNN-DMPC"]
    nrm = {a: norm_stats(a) for a in algos}
    meas = {a: raw_stats(a)["A"] for a in algos}

    modes = ["measured\n$A$", "length-controlled\n$A$", "orientation-only\n$A_{\\mathrm{dir}}$"]
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    x = np.arange(len(modes))
    w = 0.26
    for k, a in enumerate(algos):
        vals = [meas[a], nrm[a]["A_3000"], nrm[a]["Adir_3000"]]
        sds = [0.0, nrm[a]["A_3000_sd"], nrm[a]["Adir_3000_sd"]]
        ax.bar(x + (k - 1) * w, vals, width=w, yerr=sds, capsize=2.0,
               color=COLORS[a], edgecolor="black", linewidth=0.5,
               hatch=HATCHES[a], alpha=0.9, label=DISPLAY[a])
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=FS_TICK + 0.5)
    ax.set_ylabel("Alignment quality $A$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=3, loc="upper right")
    _panel(ax, "a")
    fig.tight_layout()
    fig.savefig(FIGS / "figD_normalization.pdf")
    plt.close(fig)
    for a in algos:
        print(f"  [D] {DISPLAY[a]:<22} measured={meas[a]:.4f}  "
              f"A@3000={nrm[a]['A_3000']:.4f}  Adir@3000={nrm[a]['Adir_3000']:.4f}")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    print("=== CEP figures ===")
    figA()
    figB()
    figC()
    figD()
    print("\nwritten to", FIGS)
    for f in sorted(FIGS.glob("fig[ABCD]_*.pdf")):
        print(f"  {f.name}  {f.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
