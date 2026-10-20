"""Experiment 2 (core) - how the reference direction determines the reported gain.

Why this is the core result
---------------------------
The controllers do not track the reference direction (B1/B2 track the instantaneous
optimum ``v_demand``), so the reference does not change the physical trajectory. But the
aperture gain is defined as

    aperture_gain = alignment * coverage
    coverage      = span(positions projected on v_ref) / L_ref

so coverage - and misalignment with it - is computed by projecting the array onto the
reference direction. A wrong reference makes the reported gain wrong, and both the sign
and the magnitude of that error depend on how bad the reference is.

Experiment 1 shows the reference in use is a rate-saturated greedy follower: 84% of
steps saturated, about 30% of windows collapsed. This experiment re-evaluates the same
**stored element trajectories** under

    (a) the reference in use, v_ref   - reproduces the published numbers
    (b) the optimal reference, v_opt  - the corrected one

and asks whether the ranking of the controllers changes and whether the gaps change.

Methodologically this is a **pure re-analysis**, in the same spirit as the length-
controlled re-analysis of the earlier study: no controller is re-run, only the metrics
are recomputed from ``trajectory_uav.csv``. The result is therefore fully reproducible.

Outputs
-------
exp2_metric_recompute.csv       per (site, controller, trial) detail
exp2_by_controller.csv          per (site, controller) means - manuscript Table 4
exp2_ranking.csv                ranking under both references
exp2_validation.csv             agreement with the stored metrics_summary
exp2_metric_recompute_report.md human-readable report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DT, EXP, L_REF, SIGMA, SIM, SITES, Site, load_geometry, write_json
from refgenerator import optimal_reference

# 控制器 -> 论文标签
LABEL = {
    'OLSR-FANET': 'P0', 'APF-TC': 'P1', 'GNN-TC': 'P2', 'DMPC-C': 'P3',
    'ST-GNN-DMPC': 'ST-GNN-DMPC',
    'B1_GEOM_PD': 'B1', 'B2_GREEDY': 'B2',
    'B1_GEOM_PD_PREV__w2': 'B1p', 'B2_GREEDY_PREV__w2': 'B2p',
}

# 站点 -> {报告目录: [控制器]}
SITE_DIRS = {
    'eq_pacific': {
        'scenario_c_matched': ['OLSR-FANET', 'APF-TC', 'GNN-TC', 'DMPC-C', 'ST-GNN-DMPC'],
        'scenario_c_independent_matched_full': ['B1_GEOM_PD', 'B2_GREEDY'],
        'scenario_c_independent_matched_prev': ['B1_GEOM_PD_PREV__w2', 'B2_GREEDY_PREV__w2'],
    },
    'mid30_west': {
        'multisite_mid30_west': ['OLSR-FANET', 'APF-TC', 'GNN-TC', 'DMPC-C', 'ST-GNN-DMPC',
                                 'B1_GEOM_PD', 'B2_GREEDY',
                                 'B1_GEOM_PD_PREV__w2', 'B2_GREEDY_PREV__w2'],
    },
    'mid45_europe': {
        'multisite_mid45_europe': ['OLSR-FANET', 'APF-TC', 'GNN-TC', 'DMPC-C', 'ST-GNN-DMPC',
                                   'B1_GEOM_PD', 'B2_GREEDY',
                                   'B1_GEOM_PD_PREV__w2', 'B2_GREEDY_PREV__w2'],
    },
}


# ------------------------------------------------------------------ 指标重算

def per_step_metrics(pos: np.ndarray, phi: np.ndarray, w: np.ndarray,
                     v_ref: np.ndarray, v_opt: np.ndarray,
                     valid: np.ndarray) -> pd.DataFrame:
    """逐 step 重算 aperture 指标（与 scenario_c.aperture_quality 同定义，全程向量化）。

    pos : (T, N, 2) 元素位置；phi/w : (T, M)；v_ref/v_opt : (T, 2)；valid : (T,)
    """
    T, N, _ = pos.shape
    u_hat = np.stack([np.cos(phi), np.sin(phi)], axis=-1)          # (T, M, 2)

    # alignment：各元素沿各星方位方向的投影方差
    s = np.einsum('tni,tmi->tnm', pos, u_hat)                      # (T, N, M)
    var_m = s.var(axis=1)                                          # (T, M)
    align = (w * np.exp(-var_m / max(SIGMA ** 2, 1e-9))).sum(axis=1)   # (T,)

    # coverage：阵列在参考方向上的投影跨度。
    # 注意：项目原定义带 min(1.0, ...) 截断（scenario_c.aperture_quality 第 289 行），
    # 必须逐字复现，否则与已存结果对不上（势场族会把阵列拉长到 span > L_ref）。
    r_ref = np.einsum('tni,ti->tn', pos, v_ref)
    span_ref = r_ref.max(axis=1) - r_ref.min(axis=1)
    r_opt = np.einsum('tni,ti->tn', pos, v_opt)
    span_opt = r_opt.max(axis=1) - r_opt.min(axis=1)
    cov_ref = np.minimum(1.0, span_ref / L_REF)
    cov_opt = np.minimum(1.0, span_opt / L_REF)

    # misalignment：阵列主轴与参考方向的夹角
    centered = pos - pos.mean(axis=1, keepdims=True)               # (T, N, 2)
    cov = np.einsum('tni,tnj->tij', centered, centered) / max(N, 1)  # (T, 2, 2)
    vals, vecs = np.linalg.eigh(cov)                               # (T, 2), (T, 2, 2)
    major = vecs[np.arange(T), :, np.argmax(vals, axis=1)]         # (T, 2)

    def _mis(v):
        c = np.abs(np.einsum('ti,ti->t', major, v)) / np.maximum(
            np.linalg.norm(major, axis=1), 1e-9)
        a = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
        return np.where(a > 90.0, 180.0 - a, a)

    return pd.DataFrame(dict(
        step=np.arange(T), valid=valid.astype(bool), alignment=align,
        coverage_ref=cov_ref, coverage_opt=cov_opt,
        span_ref_m=span_ref, span_opt_m=span_opt,
        gain_ref=align * cov_ref, gain_opt=align * cov_opt,
        misalign_ref_deg=_mis(v_ref), misalign_opt_deg=_mis(v_opt),
    ))


def load_intermediate(site: Site, trials: int, n_grid: int,
                      force: bool = False) -> dict:
    """计算并缓存每个 (trial) 的 v_opt 与 theta，供重算使用。"""
    cache = EXP / f'vopt_{site.name}_n{trials}_g{n_grid}.npz'
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    geo = load_geometry(site, trials)
    omega = float(geo['omega_max'][0])
    thetas, qopts = [], []
    for t in range(geo['phi'].shape[0]):
        res = optimal_reference(geo['phi'][t], geo['w'][t], geo['valid'][t],
                                omega, DT, L_REF, SIGMA, n_grid=n_grid)
        thetas.append(res.theta); qopts.append(res.q)
    out = dict(theta_opt=np.stack(thetas), q_opt=np.stack(qopts),
               omega_max=np.array([omega]), n_grid=np.array([n_grid]))
    np.savez_compressed(cache, **out)
    return out


def read_trial(tdir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """读一个 trial 的元素轨迹与已存时间序列。"""
    uav = pd.read_csv(tdir / 'trajectory_uav.csv')
    uav = uav.sort_values(['step', 'uav_id'])
    n_uav = int(uav.uav_id.nunique())
    T = int(uav.step.nunique())
    pos = uav[['x_m', 'y_m']].to_numpy(dtype=float).reshape(T, n_uav, 2)
    ts = pd.read_csv(tdir / 'metrics_timeseries.csv')
    return pos, ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--grid', type=int, default=1800)
    ap.add_argument('--sites', nargs='*', default=[s.name for s in SITES])
    args = ap.parse_args()

    reports = SIM / 'reports'
    _needed = {sub for subs in SITE_DIRS.values() for sub in subs}
    _have = [s for s in _needed if (reports / s / 'trials' / 'M3').exists()]
    if not _have:
        raise SystemExit(
            '\n[前置数据缺失] 本实验是对闭环仿真的**纯重分析**，需要先重跑那一步'
            '生成逐 step 元素轨迹：\n'
            f'  缺少目录: {reports}/<run>/trials/M3\n'
            '\n请先在本仓库根目录执行（约数分钟，10 进程）：\n'
            '  python analysis/run_scenario_c.py --mode matched --trials 20 --workers 10\n'
            '  python analysis/run_scenario_c.py --mode independent --trials 20 --workers 10\n'
            '\n说明：本仓库不随附逐 step 轨迹（约 1 GB），只提供聚合结果表；\n'
            '见 docs/REPRODUCING.md 与 reference_study/README.md。\n')
    print(f'[前置数据] 找到 {len(_have)}/{len(_needed)} 个运行目录', flush=True)

    rows, valid_rows = [], []
    for site in SITES:
        if site.name not in args.sites:
            continue
        print(f'--- {site.name} ...', flush=True)
        geo = load_geometry(site, args.trials)
        vo = load_intermediate(site, args.trials, args.grid)

        for sub, algos in SITE_DIRS[site.name].items():
            base = reports / sub / 'trials' / 'M3'
            if not base.exists():
                print(f'    [跳过] 缺少 {base}'); continue
            for algo in algos:
                adir = base / algo
                if not adir.exists():
                    continue
                for tdir in sorted(p for p in adir.iterdir() if p.is_dir()):
                    trial = int(tdir.name.rsplit('_', 1)[-1])
                    if trial >= args.trials:
                        continue
                    pos, ts = read_trial(tdir)
                    phi = geo['phi'][trial]; w = geo['w'][trial]
                    valid = geo['valid'][trial]
                    v_ref = np.stack([np.cos(geo['th_ref'][trial]),
                                      np.sin(geo['th_ref'][trial])], axis=1)
                    th = vo['theta_opt'][trial]
                    v_opt = np.stack([np.cos(th), np.sin(th)], axis=1)

                    m = per_step_metrics(pos, phi, w, v_ref, v_opt, valid)
                    mv = m[m.valid]
                    lab = LABEL.get(algo, algo)
                    rows.append(dict(
                        site=site.name, controller=algo, label=lab, trial=trial,
                        n_valid=len(mv),
                        alignment=float(mv.alignment.mean()),
                        coverage_ref=float(mv.coverage_ref.mean()),
                        coverage_opt=float(mv.coverage_opt.mean()),
                        span_ref_m=float(mv.span_ref_m.mean()),
                        span_opt_m=float(mv.span_opt_m.mean()),
                        gain_ref=float(mv.gain_ref.mean()),
                        gain_opt=float(mv.gain_opt.mean()),
                        misalign_ref_deg=float(mv.misalign_ref_deg.mean()),
                        misalign_opt_deg=float(mv.misalign_opt_deg.mean()),
                    ))
                    # 校验：与已存时间序列对照（现有参考口径）
                    valid_rows.append(dict(
                        site=site.name, controller=algos and algo, trial=trial,
                        max_abs_d_coverage=float(np.nanmax(np.abs(
                            m.coverage_ref.to_numpy() - ts.aperture_coverage.to_numpy()[:len(m)]))),
                        max_abs_d_align=float(np.nanmax(np.abs(
                            m.alignment.to_numpy() - ts.aperture_alignment.to_numpy()[:len(m)]))),
                        max_abs_d_gain=float(np.nanmax(np.abs(
                            m.gain_ref.to_numpy() - ts.aperture_gain.to_numpy()[:len(m)]))),
                        max_abs_d_mis=float(np.nanmax(np.abs(
                            m.misalign_ref_deg.to_numpy()
                            - ts.aperture_misalignment_deg.to_numpy()[:len(m)]))),
                    ))

    det = pd.DataFrame(rows)
    val = pd.DataFrame(valid_rows)
    if det.empty:
        raise SystemExit('[无结果] 未读到任何 trial 目录。请检查 --sites / --trials '
                         '与前置仿真产物（见上方说明）。')
    det['gain_delta_pct'] = (det.gain_opt / det.gain_ref - 1.0) * 100.0
    det['misalign_delta_deg'] = det.misalign_opt_deg - det.misalign_ref_deg

    EXP.mkdir(parents=True, exist_ok=True)
    det.to_csv(EXP / 'exp2_metric_recompute.csv', index=False, encoding='utf-8')
    val.to_csv(EXP / 'exp2_validation.csv', index=False, encoding='utf-8')

    agg = (det.groupby(['site', 'label'])
              .agg(n_trials=('trial', 'count'),
                   alignment=('alignment', 'mean'),
                   gain_ref=('gain_ref', 'mean'), gain_opt=('gain_opt', 'mean'),
                   coverage_ref=('coverage_ref', 'mean'), coverage_opt=('coverage_opt', 'mean'),
                   misalign_ref_deg=('misalign_ref_deg', 'mean'),
                   misalign_opt_deg=('misalign_opt_deg', 'mean'),
                   gain_delta_pct=('gain_delta_pct', 'mean'))
              .reset_index())
    agg.to_csv(EXP / 'exp2_by_controller.csv', index=False, encoding='utf-8')

    rank_rows = []
    for site, g in agg.groupby('site'):
        a = g.sort_values('gain_ref', ascending=False)['label'].tolist()
        b = g.sort_values('gain_opt', ascending=False)['label'].tolist()
        rank_rows.append(dict(site=site, ranking_ref=' > '.join(a),
                              ranking_opt=' > '.join(b), identical=bool(a == b)))
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(EXP / 'exp2_ranking.csv', index=False, encoding='utf-8')
    write_json(EXP / 'exp2_ranking.json', rank_rows)

    print('\n=== 校验（与已存时间序列对照，现有参考口径）===')
    for c in ['max_abs_d_coverage', 'max_abs_d_align', 'max_abs_d_gain', 'max_abs_d_mis']:
        print(f'  {c:24s}: max = {val[c].max():.3e}')
    print('\n=== 逐控制器（赤道）===')
    sub = agg[agg.site == 'eq_pacific'].sort_values('gain_ref', ascending=False)
    print(sub.round(4).to_string(index=False))
    print('\n=== 排序对照 ===')
    print(rk.to_string(index=False))

    _report(agg, rk, val)


def _report(agg: pd.DataFrame, rk: pd.DataFrame, val: pd.DataFrame) -> None:
    L = []
    A = L.append
    A('# 实验二 · 参考方向决定被报道的孔径增益\n')
    A('> 脚本 `01_code/exp2_metric_recompute.py`；明细 `exp2_metric_recompute.csv`；'
      '汇总 `exp2_by_controller.csv`；排序 `exp2_ranking.csv`\n')
    A('## 1. 为什么参考方向会影响"增益"\n')
    A('控制器**不跟踪参考方向**（B1/B2 跟的是瞬时最优 `v_demand`），所以参考方向不改变物理轨迹。')
    A('但项目的增益定义是\n')
    A('```')
    A('aperture_gain = alignment * coverage')
    A('coverage      = span(阵列在参考方向上的投影) / L_ref')
    A('```\n')
    A('**coverage 与 misalignment 都是在参考方向上量的**。参考错了，报道的增益就跟着错。')
    A('本实验用同一批已存元素轨迹（`trajectory_uav.csv`），只换参考方向重算 —— 纯重分析，不重跑控制器。\n')
    A('## 2. 与已存结果的一致性校验（现有参考口径）\n')
    A('| 量 | 逐 step 最大绝对偏差 |')
    A('|:--|--:|')
    for k, name in [('max_abs_d_align', 'alignment'), ('max_abs_d_coverage', 'coverage'),
                    ('max_abs_d_gain', 'gain'), ('max_abs_d_mis', 'misalignment')]:
        A(f"| {name} | {val[k].max():.3e} |")
    A('')
    A('## 3. 逐控制器结果（现有参考 vs 最优参考）\n')
    for site, g in agg.groupby('site'):
        A(f'### {site}\n')
        A('| 控制器 | G (v_ref) | G (v_opt) | 变化 | misalign (v_ref) | misalign (v_opt) |')
        A('|:--|--:|--:|--:|--:|--:|')
        for _, r in g.sort_values('gain_ref', ascending=False).iterrows():
            A(f"| {r['label']} | {r['gain_ref']:.4f} | {r['gain_opt']:.4f} | "
              f"{r['gain_delta_pct']:+.1f}% | {r['misalign_ref_deg']:.2f}° | "
              f"{r['misalign_opt_deg']:.2f}° |")
        A('')
    A('## 4. 控制器排序是否改变\n')
    A('| 站点 | 现有参考下的排序 | 最优参考下的排序 | 是否相同 |')
    A('|:--|:--|:--|:--:|')
    for _, r in rk.iterrows():
        A(f"| {r['site']} | {r['ranking_ref']} | {r['ranking_opt']} | "
          f"{'相同' if r['identical'] else '**不同**'} |")
    A('')
    (EXP / 'exp2_metric_recompute_report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
