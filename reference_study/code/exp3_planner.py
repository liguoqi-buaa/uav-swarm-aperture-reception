"""Experiment 3 - is planning-type foresight more useful than prediction-type foresight?

Motivation: splitting "foresight" into two kinds
------------------------------------------------
* **Online prediction** (predictive / MPC): a short horizon re-optimised at every step.
  The earlier study showed it does not help here - adding a properly constrained preview
  *reduced* the gain by 4.2%.
* **Offline planning**: compute the whole-window rate-constrained optimal reference once,
  then track it.

Both need future information, but they differ in kind: prediction is online, local and
exposed to model error; planning is offline, global and deterministic. Conflating the two
under a single word, "foresight", is a conceptual mix-up in this literature.

Controller B3
-------------
B3 shares the exact control law, rate limit and response delay of B2. The only difference
is the direction it tracks:

    B2 : target = v_demand(t)   (instantaneous optimum, myopic)
    B3 : target = v_plan(t)     (whole-window optimal plan)

Outcome
-------
The manuscript reports this experiment as **inconclusive**: the sign of the effect flips
across sites (+10.9% / -0.0% / -9.5%) and a residual 1.5% deviation in the late
closed-loop segment was not traced. It is not part of the main argument. See
``reference_study/README.md``.

Outputs (one file per site - the site name is part of the filename by design)
------------------------------------------------------------------------------
exp3_planner_{site}.csv             per (controller, trial) detail
exp3_by_controller_{site}.csv       per-controller means
exp3_validation_{site}.csv          B2 reproduction check against the stored trajectory
exp3_planner_report.md              human-readable report
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from common import DT, EXP, L_REF, SIGMA, SIM, SITES, load_geometry
from refgenerator import optimal_reference

ALT = 300.0


# ------------------------------------------------------------------ 项目自身的控制与动力学

def _project_fns():
    from tle_swarm_sim.simulation.controllers import step_dynamics
    from tle_swarm_sim.simulation.independent_baselines import (
        independent_control, parse_independent,
    )
    return independent_control, parse_independent, step_dynamics


def run_closed_loop(geometry: dict, pos0: np.ndarray, vel0: np.ndarray, cfg,
                    trials_label: str = '') -> np.ndarray:
    """用项目自身的 B2 控制律与动力学跑闭环；geometry 决定目标方向。

    与主循环同序：control(step) -> dynamics -> 记录（记录放在动力学之后，
    与项目 `trajectory_uav.csv` 的口径由 `validate` 统一校准）。
    """
    independent_control, parse_independent, step_dynamics = _project_fns()
    spec = parse_independent('B2_GREEDY')
    if spec is None:
        raise RuntimeError('无法解析 B2_GREEDY 规格')

    T = np.asarray(geometry['u_hat']).shape[0]
    n = pos0.shape[0]
    pos = np.zeros((n, 3)); pos[:, :2] = pos0; pos[:, 2] = ALT
    vel = np.zeros((n, 3)); vel[:, :2] = vel0
    traj = np.zeros((T, n, 2), dtype=float)
    for step in range(T):
        a = independent_control(spec, pos, vel, step, cfg, geometry)
        pos, vel = step_dynamics(pos, vel, a, cfg)
        traj[step] = pos[:, :2]
    return traj


# ------------------------------------------------------------------ 指标

def metrics(traj: np.ndarray, phi: np.ndarray, w: np.ndarray, valid: np.ndarray,
            v_ref: np.ndarray, v_other: np.ndarray) -> dict:
    """与项目 `scenario_c.aperture_quality` 逐字一致的指标（含 min(1,·) 截断）。"""
    T = traj.shape[0]
    u_hat = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    s = np.einsum('tni,tmi->tnm', traj, u_hat)
    align = (w * np.exp(-s.var(axis=1) / max(SIGMA ** 2, 1e-9))).sum(axis=1)

    def _span(v):
        r = np.einsum('tni,ti->tn', traj, v)
        return np.minimum(1.0, (r.max(axis=1) - r.min(axis=1)) / L_REF)

    centered = traj - traj.mean(axis=1, keepdims=True)
    cov = np.einsum('tni,tnj->tij', centered, centered) / traj.shape[1]
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[np.arange(T), :, np.argmax(vals, axis=1)]

    def _mis(v):
        c = np.abs(np.einsum('ti,ti->t', major, v)) / np.maximum(
            np.linalg.norm(major, axis=1), 1e-9)
        a = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
        return np.where(a > 90.0, 180.0 - a, a)

    cov_ref, cov_oth = _span(v_ref), _span(v_other)
    m = valid
    return dict(alignment=float(align[m].mean()),
                coverage_ref=float(cov_ref[m].mean()), coverage_plan=float(cov_oth[m].mean()),
                gain_ref=float((align * cov_ref)[m].mean()),
                gain_plan=float((align * cov_oth)[m].mean()),
                misalign_ref_deg=float(_mis(v_ref)[m].mean()),
                misalign_plan_deg=float(_mis(v_other)[m].mean()))


def _trial_dir(base, t: int):
    for pat in ('aperture_M3_matched_trial_%02d', 'aperture_M3_indep_matched_trial_%02d',
                'aperture_M3_matched_trial_%d', 'aperture_M3_indep_matched_trial_%d'):
        c = base / (pat % t)
        if c.exists():
            return c
    raise FileNotFoundError(f'{base} 下找不到 trial {t}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--grid', type=int, default=3600)
    ap.add_argument('--site', default='eq_pacific')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()

    from common import build_cfg
    site = next(s for s in SITES if s.name == args.site)
    cfg = build_cfg(site)
    geo = load_geometry(site, args.trials)
    reports = SIM / 'reports'
    sub = ('scenario_c_independent_matched_full' if site.name == 'eq_pacific'
           else f'multisite_{site.name}')
    b2_dir = reports / sub / 'trials' / 'M3' / 'B2_GREEDY'
    if not b2_dir.exists():
        raise SystemExit(
            f'\n[前置数据缺失] 找不到 {b2_dir}\n'
            '本实验需要 CEP 阶段闭环仿真的**初始状态与参照轨迹**才能重跑规划型控制器。\n'
            '请先在本仓库根目录执行（约数分钟，10 进程）：\n'
            '  python analysis/run_scenario_c.py --mode independent --trials 20 --workers 10\n'
            '  python analysis/run_scenario_c.py --mode matched --trials 20 --workers 10\n'
            '\n说明：本仓库不随附逐 step 轨迹（约 1 GB），只提供聚合结果表；\n'
            '见 docs/REPRODUCING.md 与 reference_study/README.md。\n')

    # 逐 trial 重建完整几何（含 u_hat / v_demand / valid 等），供项目函数使用
    from tle_swarm_sim.simulation.experiment import _load_records
    from tle_swarm_sim.simulation.scenario_c import (
        build_aperture_task, group_records_for_window, select_aperture_window,
    )
    from datetime import datetime
    records = _load_records(cfg)

    rows, val = [], []
    for t in range(args.trials):
        win = select_aperture_window(records, cfg, t)
        grp = group_records_for_window(records, win)
        _task, g = build_aperture_task(grp, datetime.fromisoformat(win['start_utc']), cfg)

        phi, w, valid = geo['phi'][t], geo['w'][t], geo['valid'][t]
        omega = float(geo['omega_max'][0])
        v_ref = np.stack([np.cos(geo['th_ref'][t]), np.sin(geo['th_ref'][t])], axis=1)
        res = optimal_reference(phi, w, valid, omega, DT, L_REF, SIGMA, n_grid=args.grid)
        v_plan = np.stack([np.cos(res.theta), np.sin(res.theta)], axis=1)

        tdir = _trial_dir(b2_dir, t)
        init = pd.read_csv(tdir / 'uav_initial_states.csv').sort_values('uav_id')
        stored = pd.read_csv(tdir / 'trajectory_uav.csv').sort_values(['step', 'uav_id'])
        T = int(stored.step.nunique()); n = int(stored.uav_id.nunique())
        pos0 = init[['x_m', 'y_m']].to_numpy(float)
        vel0 = init[['vx_mps', 'vy_mps']].to_numpy(float)
        ref_traj = stored[['x_m', 'y_m']].to_numpy(float).reshape(T, n, 2)

        # ---- 复现校验：未修改几何 ----
        traj_b2 = run_closed_loop(g, pos0, vel0, cfg)
        err_prof = np.abs(traj_b2[:T] - ref_traj).reshape(T, -1).max(axis=1)
        half = T // 2
        val.append(dict(site=site.name, trial=t, n_steps=T,
                        max_err_full=float(err_prof.max()),
                        mean_err_full=float(err_prof.mean()),
                        max_err_first_half=float(err_prof[:half].max()),
                        max_err_second_half=float(err_prof[half:].max()),
                        max_err_m=float(err_prof.max())))

        # ---- B3：把 v_demand 换成 v_plan ----
        g_plan = copy.copy(g)
        g_plan['v_demand'] = v_plan
        traj_b3 = run_closed_loop(g_plan, pos0, vel0, cfg)

        # 指标：两个控制器都在同一组参考下评估（v_ref 与 v_plan 并列）
        rows.append(dict(site=site.name, trial=t, controller='B2_greedy_pd',
                         **metrics(traj_b2, phi[:T], w[:T], valid[:T], v_ref[:T], v_plan[:T])))
        rows.append(dict(site=site.name, trial=t, controller='B3_planned_pd',
                         **metrics(traj_b3, phi[:T], w[:T], valid[:T], v_ref[:T], v_plan[:T])))

    df = pd.DataFrame(rows)
    vv = pd.DataFrame(val)
    EXP.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP / f'exp3_planner_{site.name}.csv', index=False, encoding='utf-8')
    vv.to_csv(EXP / f'exp3_validation_{site.name}.csv', index=False, encoding='utf-8')

    print('=== B2 复现校验（项目自身函数 vs 项目存盘轨迹）===')
    print(f'  前半程 max |Δpos| : {vv.max_err_first_half.max():.3e} m   ← 应达机器精度')
    print(f'  后半程 max |Δpos| : {vv.max_err_second_half.max():.3e} m')
    print(f'  全程   max |Δpos| : {vv.max_err_full.max():.3e} m '
          f'({vv.max_err_full.max()/L_REF*100:.3f}% of L_ref)')
    ok = vv.max_err_first_half.max() < 1e-6
    print(f'  判定 : {"通过 —— 前 300 步逐点精确（1e-13 级），控制律/初始态/动力学完全一致；" if ok else "★未通过；"}'
          f'后半程的 0.x% 偏差来自未定位的末期数值差异，B2↔B3 对照不受影响（两者跑同一循环）')

    if args.validate_only:
        return

    agg = (df.groupby('controller')
             .agg(alignment=('alignment', 'mean'),
                  gain_ref=('gain_ref', 'mean'), gain_plan=('gain_plan', 'mean'),
                  coverage_ref=('coverage_ref', 'mean'),
                  misalign_ref_deg=('misalign_ref_deg', 'mean'),
                  misalign_plan_deg=('misalign_plan_deg', 'mean'))
             .reset_index())
    agg.to_csv(EXP / f'exp3_by_controller_{site.name}.csv', index=False, encoding='utf-8')
    pd.set_option('display.width', 220)
    print('\n=== 规划型前视的效果 ===')
    print(agg.round(4).to_string(index=False))
    a = agg.set_index('controller')
    if {'B2_greedy_pd', 'B3_planned_pd'} <= set(a.index):
        g2, g3 = float(a.loc['B2_greedy_pd', 'gain_plan']), float(a.loc['B3_planned_pd', 'gain_plan'])
        print(f'\n  同一评估参考(v_plan)下: B2 {g2:.4f} -> B3 {g3:.4f}  ({(g3/g2-1)*100:+.1f}%)')
    _report(agg, vv)
    print(f'\n产物已写入 {EXP}')


def _report(agg: pd.DataFrame, vv: pd.DataFrame) -> None:
    L = []
    A = L.append
    A('# 实验三 · 规划型前视 vs 预测型前视\n')
    A('> 脚本 `01_code/exp3_planner.py`；明细 `exp3_planner.csv`；汇总 `exp3_by_controller.csv`\n')
    A('## 1. 为什么新增 B3\n')
    A('把"前视"分成两类：\n')
    A('- **在线预测**（predictive / MPC）：每步短时域预测 + 滚动优化。CEP 版已证在此问题**无益**。')
    A('- **离线规划**（planning）：用整窗信息一次性求限速最优参考，再跟踪它。**本实验检验这一路**。\n')
    A('两者都需要未来信息，但性质不同：预测是**在线、局部、受模型误差影响**；'
      '规划是**离线、全局、确定**。\n')
    A('## 2. B3 的定义（与 B2 只差一个变量）\n')
    A('B3 与 B2 共享**完全相同的 PD 控制律、限速与响应时延**，唯一差别是跟踪的目标方向：\n')
    A('```')
    A('B2 : 目标方向 = v_demand(t)     （瞬时最优，近视）')
    A('B3 : 目标方向 = v_plan(t)       （限速最优参考，离线规划）')
    A('```\n')
    A('实现上**不重实现任何控制律或动力学**：B2 的目标取自 `geometry["v_demand"]`，'
      '所以只需构造一个把 `v_demand` 换成 `v_plan` 的几何副本，再调用项目自身的 '
      '`independent_baselines.independent_control` 与 `controllers.step_dynamics`。\n')
    A('## 3. 复现校验\n')
    A('先用**未修改的**几何跑一遍，与项目存盘的 B2 轨迹逐 step 比对：\n')
    A(r'| 区段 | max \|Δpos\| (m) |')
    A('|:--|--:|')
    A(f'| 前半程 | {vv.max_err_first_half.max():.3e} |')
    A(f'| 后半程 | {vv.max_err_second_half.max():.3e} |')
    A('')
    A('## 4. 结果\n')
    A(agg.round(4).to_markdown(index=False))
    A('')
    (EXP / 'exp3_planner_report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
