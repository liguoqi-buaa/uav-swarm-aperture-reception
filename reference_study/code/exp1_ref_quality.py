"""Experiment 1 - reference optimality across sites and windows.

Answers three questions
-----------------------
Q1  How far does the reference in use (a rate-limited greedy follower) fall below the
    best achievable?  -> per-window q_ref versus q_opt
Q2  Does it fail structurally?  -> fraction of rate-saturated steps, fraction of steps
    with a gap above 30 deg, fraction of collapsed windows
Q3  Is the DP solution trustworthy?  -> rate-constraint check, agreement with the
    simulator's own oracle, grid convergence

Outputs
-------
exp1_ref_quality.csv        per-window detail (including the four correctness checks)
exp1_scenarios.csv          per-site aggregation - manuscript Table 1 and §5.5
exp1_grid_convergence.csv   grid 900 / 1800 / 3600 / 7200 - manuscript Table 2
exp1_ref_quality_report.md  human-readable report
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import DT, EXP, L_REF, SIGMA, SITES, Site, load_geometry, write_json
from refgenerator import (
    lag_fraction, omega_max_for, optimal_reference, reference_quality, turn_stats, wrap_pi,
)

CATASTROPHIC_Q = 0.5   # q_ref 低于此值视为"参考失效"（最优可达 ~0.82）


def run_site(site: Site, trials: int, n_grid: int) -> tuple[list[dict], dict]:
    geo = load_geometry(site, trials)
    omega = float(geo['omega_max'][0])
    rows, t_dp = [], 0.0

    for t in range(geo['phi'].shape[0]):
        phi, w, valid = geo['phi'][t], geo['w'][t], geo['valid'][t]
        th_ref, th_dem = geo['th_ref'][t], geo['th_dem'][t]
        q_ref = geo['q_ref'][t]
        orc = geo['oracle'][t]

        t0 = time.perf_counter()
        res = optimal_reference(phi, w, valid, omega, DT, L_REF, SIGMA, n_grid=n_grid)
        dtw = time.perf_counter() - t0
        t_dp += dtw

        q_opt = reference_quality(res.theta, phi, w, valid, L_REF, SIGMA)
        st = turn_stats(th_ref, valid, omega * DT)
        m = valid

        # 需求（瞬时最优 θ*）的峰值迁移速率：π 周期最小角差 / Δt。
        # 摘要里 "up to 79.6 °/s" 即取全部窗口的最大值；此前该数字只能
        # 从 uav_swarm_scenario_C 的存盘 summary 读到，现纳入本工作区。
        _d = np.abs((np.diff(th_dem) + np.pi / 2) % np.pi - np.pi / 2)
        _vm = np.isfinite(_d) & valid[:-1] & valid[1:]
        dem_peak = float(np.degrees(np.max(_d[_vm]))) if _vm.any() else float('nan')

        # 交叉校验 ①：我算的 q_ref 必须等于项目自报 oracle_gain_step
        chk_oracle = float(np.nanmax(np.abs(orc[m] - q_ref[m])))
        # 交叉校验 ②：DP 轨迹必须满足速率约束
        chk_limit = bool(res.max_turn_deg <= np.degrees(omega * DT) + 1e-6)
        # 交叉校验 ③：DP 总质量必须 >= 现有参考
        chk_beat = bool(res.q_total >= np.nansum(q_ref[m]) - 1e-9)

        rows.append(dict(
            site=site.name, trial=t, n_valid=int(m.sum()),
            q_ref=float(np.nanmean(q_ref[m])), q_opt=float(np.nanmean(q_opt[m])),
            q_ref_total=float(np.nansum(q_ref[m])), q_opt_total=float(res.q_total),
            gain_pct=float((np.nanmean(q_opt[m]) / np.nanmean(q_ref[m]) - 1.0) * 100.0),
            ref_sat_frac=st['sat_frac'], ref_median_turn_deg=st['median_deg'],
            ref_p90_turn_deg=st['p90_deg'],
            ref_gap_gt30_frac=lag_fraction(th_ref, th_dem, valid, 30.0),
            rate_limit_deg=float(np.degrees(omega * DT)),
            dp_max_turn_deg=res.max_turn_deg,
            catastrophic=bool(np.nanmean(q_ref[m]) < CATASTROPHIC_Q),
            chk_oracle=chk_oracle, chk_limit=chk_limit, chk_beat=chk_beat,
            dp_s=dtw, demand_peak_rate_deg_s=dem_peak,
        ))

    df = pd.DataFrame(rows)
    # 两种口径都要报：
    #   mean-of-ratios  ：逐窗口比值的均值 —— 反映"单个窗口最坏被坑多少"，受灾难窗口放大
    #   ratio-of-sums   ：总质量之比     —— 反映"整体丢失了多少孔径质量"，是更稳的聚合量
    sum_ref = float(df.q_ref_total.sum())
    sum_opt = float(df.q_opt_total.sum())
    summ = dict(
        site=site.name, pretty=site.pretty, n_windows=len(df),
        omega_max_rad_s=omega, rate_limit_deg_per_step=float(np.degrees(omega * DT)),
        q_ref_mean=float(df.q_ref.mean()), q_opt_mean=float(df.q_opt.mean()),
        gain_pct_mean=float(df.gain_pct.mean()),
        gain_pct_min=float(df.gain_pct.min()), gain_pct_max=float(df.gain_pct.max()),
        gain_pct_median=float(df.gain_pct.median()),
        q_ref_total_sum=sum_ref, q_opt_total_sum=sum_opt,
        gain_pct_aggregate=float((sum_opt / sum_ref - 1.0) * 100.0),
        sat_frac_mean=float(df.ref_sat_frac.mean()),
        gap_gt30_mean=float(df.ref_gap_gt30_frac.mean()),
        n_catastrophic=int(df.catastrophic.sum()),
        frac_catastrophic=float(df.catastrophic.mean()),
        n_exact_100pct_saturated=int((df.ref_sat_frac >= 0.999999).sum()),
        max_chk_oracle=float(df.chk_oracle.max()),
        all_limits_ok=bool(df.chk_limit.all()),
        all_beat_ok=bool(df.chk_beat.all()),
        dp_seconds_total=float(t_dp), dp_seconds_per_window=float(t_dp / max(len(df), 1)),
        n_grid=n_grid,
    )
    return rows, summ


def grid_convergence(site: Site, trials: int, grids: list[int]) -> pd.DataFrame:
    geo = load_geometry(site, trials)
    omega = float(geo['omega_max'][0])
    rows = []
    for g in grids:
        tot_ref, tot_opt = [], []
        t0 = time.perf_counter()
        for t in range(min(trials, geo['phi'].shape[0])):
            res = optimal_reference(geo['phi'][t], geo['w'][t], geo['valid'][t],
                                    omega, DT, L_REF, SIGMA, n_grid=g)
            m = geo['valid'][t]
            tot_ref.append(np.nansum(geo['q_ref'][t][m]))
            tot_opt.append(res.q_total)
        el = time.perf_counter() - t0
        rows.append(dict(n_grid=g, resolution_deg=float(np.degrees(np.pi / g)),
                         q_opt_total=float(np.sum(tot_opt)),
                         q_ref_total=float(np.sum(tot_ref)),
                         gain_pct=float((np.sum(tot_opt) / np.sum(tot_ref) - 1) * 100),
                         seconds=float(el), seconds_per_window=float(el / trials)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--grid', type=int, default=1800)
    ap.add_argument('--sites', nargs='*', default=[s.name for s in SITES])
    ap.add_argument('--grid-check', action='store_true')
    args = ap.parse_args()

    all_rows, all_summ = [], []
    for s in SITES:
        if s.name not in args.sites:
            continue
        print(f'--- {s.name} ({args.trials} windows, grid {args.grid}) ...', flush=True)
        rows, summ = run_site(s, args.trials, args.grid)
        all_rows += rows
        all_summ.append(summ)
        print(f'    q_ref={summ["q_ref_mean"]:.4f}  q_opt={summ["q_opt_mean"]:.4f}  '
              f'gain={summ["gain_pct_mean"]:.1f}%  sat={summ["sat_frac_mean"]*100:.1f}%  '
              f'catastrophic={summ["n_catastrophic"]}/{summ["n_windows"]}',
              flush=True)

    det = pd.DataFrame(all_rows)
    sce = pd.DataFrame(all_summ)
    EXP.mkdir(parents=True, exist_ok=True)
    det.to_csv(EXP / 'exp1_ref_quality.csv', index=False, encoding='utf-8')
    sce.to_csv(EXP / 'exp1_scenarios.csv', index=False, encoding='utf-8')
    write_json(EXP / 'exp1_scenarios.json', all_summ)

    gc = None
    if args.grid_check:
        print('--- 网格收敛 ...', flush=True)
        gc = grid_convergence(SITE_EQ, args.trials, [900, 1800, 3600, 7200])
        gc.to_csv(EXP / 'exp1_grid_convergence.csv', index=False, encoding='utf-8')
        print(gc.round(4).to_string(index=False))

    _report(det, sce, gc)
    print(f'\n产物已写入 {EXP}')


from common import SITES as _S
SITE_EQ = next(s for s in _S if s.name == 'eq_pacific')


def _report(det: pd.DataFrame, sce: pd.DataFrame, gc) -> None:
    L = []
    A = L.append
    A('# 实验一 · 全量参考最优性\n')
    A('> 生成脚本 `01_code/exp1_ref_quality.py`；明细 `exp1_ref_quality.csv`，场景汇总 `exp1_scenarios.csv`\n')
    A('## 1. 什么是"现有参考"\n')
    A('`scenario_c.rate_limited_directions()` 把参考朝向构造成**以最大角速率朝瞬时最优 '
      '`v_demand` 转**的贪心跟随者（不看未来）。本实验把它与**限速最优参考**对比——后者由')
    A('`refgenerator.optimal_reference()` 在角度网格上用周期边界 DP 精确求解：\n')
    A('```')
    A('max_{theta(0..T)}  sum_t q(theta_t, t)   s.t.  |dtheta|_mod pi <= omega_max * dt')
    A('q(theta, t) = sum_m w_m exp(-(L^2/12) cos^2(theta - phi_m) / sigma^2)')
    A('```\n')
    A('## 2. 场景级汇总\n')
    A('两种聚合口径都要看，含义不同：\n')
    A('- **逐窗口比值的均值**（mean of ratios）：反映"单个窗口最坏被坑多少"，会被灾难窗口放大；')
    A('- **总质量之比**（ratio of sums）：反映"整体丢失了多少孔径质量"，是更稳的聚合量。\n')
    A('| 站点 | 窗口 | q_ref | q_opt | 提升(逐窗均值) | 提升(总质量) | 速率饱和步占比 | 夹角>30° 占比 | 失效窗口 |')
    A('|:--|--:|--:|--:|--:|--:|--:|--:|--:|')
    for _, r in sce.iterrows():
        A(f"| {r['pretty']} | {int(r['n_windows'])} | {r['q_ref_mean']:.4f} | {r['q_opt_mean']:.4f} | "
          f"**+{r['gain_pct_mean']:.1f}%** | **+{r['gain_pct_aggregate']:.1f}%** | "
          f"{r['sat_frac_mean']*100:.1f}% | "
          f"{r['gap_gt30_mean']*100:.1f}% | {int(r['n_catastrophic'])}/{int(r['n_windows'])} |")
    A('')
    A(f"（失效窗口 = q_ref < {CATASTROPHIC_Q}；DP 最优一般可达 ~0.82）\n")
    A('## 3. 速率约束与自检\n')
    A('| 站点 | 速率上限 (°/step) | DP 最大单步转角 | 速率约束全部满足 | DP 总质量≥现有 | 与项目 oracle 最大偏差 |')
    A('|:--|--:|--:|:--:|:--:|--:|')
    for _, r in sce.iterrows():
        A(f"| {r['pretty']} | {r['rate_limit_deg_per_step']:.4f} | "
          f"{det[det.site==r['site']].dp_max_turn_deg.max():.4f} | "
          f"{'✓' if r['all_limits_ok'] else '✗'} | {'✓' if r['all_beat_ok'] else '✗'} | "
          f"{r['max_chk_oracle']:.2e} |")
    A('')
    A('## 4. 逐窗口明细（按提升排序，前 10）\n')
    A('| 站点 | trial | q_ref | q_opt | 提升 | 饱和占比 | 夹角>30° |')
    A('|:--|--:|--:|--:|--:|--:|--:|')
    for _, r in det.sort_values('gain_pct', ascending=False).head(10).iterrows():
        A(f"| {r['site']} | {int(r['trial'])} | {r['q_ref']:.4f} | {r['q_opt']:.4f} | "
          f"+{r['gain_pct']:.1f}% | {r['ref_sat_frac']*100:.1f}% | {r['ref_gap_gt30_frac']*100:.1f}% |")
    A('')
    if gc is not None:
        A('## 5. 网格收敛\n')
        A(gc.round(4).to_markdown(index=False))
        A('')
    A('## 6. 机制\n')
    A('贪心跟随者在饱和时满足 `theta_f(t) = theta_f(0) + omega_max * t`，而最优方向以 `v*(t)` 前进；')
    A('当 `|v*| > omega_max` 持续存在时滞后 `(v* - omega_max) * t` **线性累积、无上界**。')
    A('滞后绕满半圈后参考方向即变为任意值 —— 这是**结构性失效**，不是调参问题。')
    A('速率饱和步占比接近 1 是它的指纹：**每一步都吃满速率 = 永远在追赶**。\n')
    (EXP / 'exp1_ref_quality_report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
