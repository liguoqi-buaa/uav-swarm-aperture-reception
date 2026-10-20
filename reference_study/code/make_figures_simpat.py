"""Regenerate Figures 1-3 (single-column, vertically stacked panels).

Why the panels are stacked
--------------------------
The target journal is two-column and one column is about 3.2 in wide. The first version of
these figures used a 7.0 in width; once typeset to the column width in a single column
the scale factor was about 0.46, so 8 pt text rendered at roughly 3.7 pt. The width here is set to the column width (3.25 in), which makes the scale factor
about 1.0 and the point sizes what they claim to be. The panels are stacked accordingly.

    fig1_reference_failure.pdf   (a) orientation trajectories in one window
                                 (b) aperture quality over the same window
    fig2_reference_quality.pdf   (a) per-window q_ref versus q_opt
                                 (b) saturated-step fraction and the gain from correcting
    fig3_gain_shift.pdf          aperture gain under both references, one panel per site

Data come from ``../experiments/`` - the same files behind the manuscript tables. No value
is recomputed or adjusted here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DT, EXP, L_REF, SIGMA, SITES, load_geometry
from refgenerator import optimal_reference, quality_on_grid

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'figures'
if not FIG.exists():
    FIG = HERE.parent / '03_figures'
FIG.mkdir(parents=True, exist_ok=True)

# cas-dc 单栏宽度；宽度等于栏宽 ⇒ 缩放比 ≈ 1，字号所见即所得
COL_W = 3.25

plt.rcParams.update({
    'font.size': 7.5, 'axes.linewidth': 0.7, 'lines.linewidth': 1.0,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
})

GREY = '#8a8a8a'
RED = '#c1272d'
BLUE = '#1f4e79'


# ---------------------------------------------------------------- Fig 1
def fig1(site_name='eq_pacific', trials=20):
    site = next(s for s in SITES if s.name == site_name)
    geo = load_geometry(site, trials)
    det = pd.read_csv(EXP / 'exp1_ref_quality.csv')
    sub = det[det.site == site_name]
    t = int(sub.sort_values('gain_pct', ascending=False).trial.iloc[0])   # 最坏窗口

    phi, w, valid = geo['phi'][t], geo['w'][t], geo['valid'][t]
    th_ref, th_dem = geo['th_ref'][t], geo['th_dem'][t]
    omega = float(geo['omega_max'][0])
    res = optimal_reference(phi, w, valid, omega, DT, L_REF, SIGMA, n_grid=3600)
    th_opt = res.theta
    T = len(th_ref)
    time = np.arange(T) * DT

    q_ref = geo['q_ref'][t]
    q_opt = np.array([quality_on_grid(np.array([th_opt[i]]), phi[i], w[i], L_REF, SIGMA)[0]
                      for i in range(T)])

    fig, ax = plt.subplots(2, 1, figsize=(COL_W, 4.75))

    def fold(a):
        d = np.degrees(np.mod(a, np.pi))
        return np.where(d > 90, d - 180, d)

    # (a) 朝向轨迹 —— 竖排后宽度小，图例标签缩短
    ax[0].plot(time, fold(th_dem), color=GREY, lw=0.9,
               label=r'optimum $\theta^\star(t)$')
    ax[0].plot(time, fold(th_ref), color=RED, lw=1.2,
               label=r'reference in use ($v_{\rm ref}$)')
    ax[0].plot(time, fold(th_opt), color=BLUE, lw=1.2, ls='--',
               label=r'optimal ($v_{\rm opt}$)')
    # 顶上留出图例空间，避免像横排版那样压住轨迹
    ax[0].set_ylim(-100, 152)
    ax[0].set_xlabel('time (s)', labelpad=1.5)
    ax[0].set_ylabel('array orientation\n(deg, folded)', labelpad=1.5)
    ax[0].set_title('(a)  The reference never catches up', fontsize=7.2, pad=3)
    ax[0].legend(loc='upper left', fontsize=5.6, handlelength=1.5,
                 borderpad=0.2, labelspacing=0.22)

    # (b) 质量曲线
    ax[1].plot(time, q_opt, color=BLUE, lw=1.2, label='achievable (optimal)')
    ax[1].plot(time, q_ref, color=RED, lw=1.2, label='achieved (in use)')
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel('time (s)', labelpad=1.5)
    ax[1].set_ylabel(r'aperture quality $q(\theta,t)$', labelpad=1.5)
    # 均值并入标题第二行：轴内不留文本，就不会压到曲线
    ax[1].set_title(
        '(b)  Reference quality collapses\n'
        f'mean $q$: {np.nanmean(q_ref):.3f} $\\rightarrow$ {np.nanmean(q_opt):.3f}'
        f'  ({np.nanmean(q_opt)/np.nanmean(q_ref)-1:+.0%})',
        fontsize=6.9, pad=3, linespacing=1.35)
    # 白底图例：右下角红线有爬升，不透明底比裸线更易读
    ax[1].legend(loc='lower right', fontsize=5.6, handlelength=1.5,
                 borderpad=0.25, labelspacing=0.22,
                 frameon=True, framealpha=0.9, edgecolor='none')

    fig.subplots_adjust(hspace=0.42)
    fig.savefig(FIG / 'fig1_reference_failure.pdf')
    plt.close(fig)
    print('fig1 OK  (site=%s, worst trial=%d)' % (site_name, t))


# ---------------------------------------------------------------- Fig 2
def fig2():
    det = pd.read_csv(EXP / 'exp1_ref_quality.csv')
    sce = pd.read_csv(EXP / 'exp1_scenarios.csv')
    order = ['eq_pacific', 'mid30_west', 'mid45_europe']
    nice = {'eq_pacific': 'equatorial\n0.0$^{\\circ}$N',
            'mid30_west': 'mid-lat.\n30$^{\\circ}$N',
            'mid45_europe': 'mid-lat.\n45$^{\\circ}$N'}

    fig, ax = plt.subplots(2, 1, figsize=(COL_W, 4.95))

    # (a) 逐窗口 q_ref vs q_opt（按 q_opt 排序）
    pos = 0
    ticks, labels = [], []
    for i, s in enumerate(order):
        g = det[det.site == s].sort_values('q_opt')
        x = np.arange(len(g)) + pos
        ax[0].plot(x, g.q_opt, color=BLUE, lw=0.8, marker='.', ms=1.8, ls='-',
                   label='optimal reference' if i == 0 else None)
        ax[0].plot(x, g.q_ref, color=RED, lw=0.8, marker='.', ms=1.8, ls='-',
                   label='reference in use' if i == 0 else None)
        ax[0].axhline(0.5, color=GREY, lw=0.6, ls=':')
        ticks.append(pos + len(g) / 2); labels.append(nice[s])
        pos += len(g) + 4
    ax[0].set_xticks(ticks); ax[0].set_xticklabels(labels, fontsize=6.2)
    ax[0].set_ylabel('reference quality $q$', labelpad=1.5)
    ax[0].set_title('(a)  Per-window reference quality (sorted)', fontsize=7.5, pad=3)
    ax[0].legend(loc='lower right', fontsize=5.8, handlelength=1.6,
                 borderpad=0.2, labelspacing=0.25)

    # (b) 饱和占比 vs 提升（双 y 轴）
    ax2 = ax[1]
    x = np.arange(len(order))
    sat = [float(sce[sce.site == s].sat_frac_mean.iloc[0]) * 100 for s in order]
    gain = [float(sce[sce.site == s].gain_pct_aggregate.iloc[0]) for s in order]
    ax2.bar(x - 0.19, sat, width=0.36, color=RED, alpha=0.85,
            label='steps at the rate limit')
    ax2.set_ylabel('rate-saturated\nsteps (%)', color=RED, labelpad=1.5)
    ax2.set_ylim(0, 112)
    ax2.tick_params(axis='y', labelcolor=RED, labelsize=6.8)
    ax2.set_xticks(x); ax2.set_xticklabels([nice[s] for s in order], fontsize=6.2)
    ax3 = ax2.twinx()
    ax3.bar(x + 0.19, gain, width=0.36, color=BLUE, alpha=0.85,
            label='quality gain from correcting it')
    ax3.set_ylabel('quality gain (%)', color=BLUE, labelpad=1.5)
    ax3.tick_params(axis='y', labelcolor=BLUE, labelsize=6.8)
    ax3.grid(False)
    ax3.set_ylim(0, max(gain) * 1.55)
    ax2.set_title('(b)  Saturation and the gain from correcting it', fontsize=7.5, pad=3)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=5.7,
               handlelength=1.4, borderpad=0.2, labelspacing=0.2, ncol=1)

    fig.subplots_adjust(hspace=0.45)
    fig.savefig(FIG / 'fig2_reference_quality.pdf')
    plt.close(fig)
    print('fig2 OK')


# ---------------------------------------------------------------- Fig 3
def fig3():
    agg = pd.read_csv(EXP / 'exp2_by_controller.csv')
    order_lab = ['B2', 'B2p', 'B1', 'B1p', 'ST-GNN-DMPC', 'P3', 'P2', 'P1', 'P0']
    site3 = ['eq_pacific', 'mid30_west', 'mid45_europe']
    nice = {'eq_pacific': '0.0$^{\\circ}$N', 'mid30_west': '30$^{\\circ}$N',
            'mid45_europe': '45$^{\\circ}$N'}

    fig, ax = plt.subplots(3, 1, figsize=(COL_W, 6.35), sharey=True)
    for k, s in enumerate(site3):
        g = agg[agg.site == s].set_index('label').reindex(order_lab)
        x = np.arange(len(order_lab))
        ax[k].bar(x - 0.20, g.gain_ref, width=0.40, color=GREY,
                  label=r'$v_{\rm ref}$ (in use)')
        ax[k].bar(x + 0.20, g.gain_opt, width=0.40, color=BLUE,
                  label=r'$v_{\rm opt}$ (corrected)')
        ax[k].set_xticks(x)
        # 只在最下方子图显示控制器名（上方两幅共享同一横轴），旋转 90° 最省横向空间
        if k == len(site3) - 1:
            ax[k].set_xticklabels(order_lab, rotation=90, fontsize=5.8)
        else:
            ax[k].set_xticklabels([])
        ax[k].set_ylim(0, 0.95)
        ax[k].axvline(3.5, color='#999999', lw=0.8, ls=(0, (4, 3)), zorder=3)
        ax[k].tick_params(axis='y', labelsize=6.6)
        # 站点名放到子图内左上（不用 title，避免与族名/图例抢轴外空间）
        ax[k].text(0.015, 0.94, nice[s], transform=ax[k].transAxes,
                   fontsize=6.8, va='top', ha='left', color='#111111')

    ax[0].set_ylabel('aperture gain $G$', labelpad=1.5)
    ax[0].legend(loc='upper right', fontsize=5.6, handlelength=1.4,
                 borderpad=0.2, labelspacing=0.22)
    # 族名标注放在图例下沿（柱高上限 0.57，0.66 一线仍是空白）
    ax[0].text(1.75, 0.66, 'geometric', ha='center', fontsize=5.2, color='#444444')
    ax[0].text(6.10, 0.66, 'predictive / parametric', ha='center',
               fontsize=5.2, color='#444444')
    fig.subplots_adjust(hspace=0.20)
    fig.savefig(FIG / 'fig3_gain_shift.pdf')
    plt.close(fig)
    print('fig3 OK')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    print('插图已写入', FIG)
