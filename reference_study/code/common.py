"""Shared utilities: site definitions, SimulationConfig construction, geometry cache.

Design rules
------------
* **Read-only on the simulator.** Nothing under ``uav_swarm_scenario_C/`` is modified;
  its geometry construction is imported only.
* **Same protocol as the earlier study.** Sites, window pools, aperture parameters and
  the elevation threshold are reused verbatim, so old and new results are comparable.
* **Geometry cache.** SGP4 propagation is the bottleneck, so each site is computed once
  and stored as npz for reuse by exp1 / exp2 / exp3.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                  # 本目录的上一级（含 code/ experiments/ figures/）
REPO = ROOT.parent                  # 仓库根

SIM = REPO / 'uav_swarm_scenario_C'  # 仿真器（本仓库自带）

# 窗口池目录：公开仓库里在 analysis/，作者本机工作区里在 _c_work/。
# 自动探测，两种情况都能跑，无需手工改路径。
for _cand in (REPO / 'analysis', REPO / '_c_work'):
    if (_cand / 'covisible_pool.json').exists():
        WORK = _cand
        break
else:
    WORK = REPO / '_c_work'

# 实验产物目录：公开仓库里是 experiments/，本机工作区里是 02_experiments/
EXP = ROOT / 'experiments'
if not EXP.exists():
    _alt = ROOT / '02_experiments'
    if _alt.exists():
        EXP = _alt

for p in (str(SIM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 与 CEP 版逐字一致的孔径参数
L_REF = 3000.0
SIGMA = 500.0
MIN_ELEV_DEG = 10.0
DURATION_S = 600.0
DT = 1.0
MAX_VEL_MPS = 20.0


@dataclass(frozen=True, slots=True)
class Site:
    name: str
    pretty: str
    lat: float
    lon: float
    pool: Path

    @property
    def tag(self) -> str:
        return self.name


SITES = (
    Site('eq_pacific', 'equatorial Pacific (0.0N, 174.62W)', 0.0, -174.624885,
         WORK / 'covisible_pool.json'),
    Site('mid30_west', 'mid-latitude 30N (30.0N, 100.0W)', 30.0, -100.0,
         WORK / 'pools' / 'covisible_pool_mid30_west.json'),
    Site('mid45_europe', 'mid-latitude 45N (45.0N, 8.0E)', 45.0, 8.0,
         WORK / 'pools' / 'covisible_pool_mid45_europe.json'),
)

SITE_BY_NAME = {s.name: s for s in SITES}


def build_cfg(site: Site):
    """构造与 CEP 版完全一致的 SimulationConfig。"""
    from tle_swarm_sim.simulation.config import SimulationConfig
    return SimulationConfig(
        repo_root=SIM, reports_dir=SIM / 'reports', figures_dir=SIM / 'reports' / 'figures',
        trials=20, concurrent_satellites=(3,),
        simulation_duration_s=DURATION_S, control_step_s=DT,
        scenario_mode='aperture',
        site_lat_deg=site.lat, site_lon_deg=site.lon,
        aperture_pool_path=site.pool,
        aperture_baseline_m=L_REF, aperture_sigma_m=SIGMA,
        aperture_min_elevation_deg=MIN_ELEV_DEG,
        monitoring_kernel_radius_m=1000.0, deployment_mode='monitor_area_near',
        monitor_area_initial_spread_m=1500.0, slot_smoothing_alpha=1.0,
        max_communication_range_m=2500.0, task_window_mode='extended_pass_10deg',
        stgnn_connectivity_blend_single_satellite=0.0, profile_gain_mode='matched',
    ).resolve()


def load_geometry(site: Site, trials: int, force: bool = False) -> dict:
    """加载（或构造并缓存）一个站点的全部窗口几何。

    返回 dict，含逐 trial 的数组堆叠：
      phi (N,T,M) 方位角 · w (N,T,M) 优先级 · valid (N,T)
      th_ref (N,T) 现有参考朝向 · th_dem (N,T) 瞬时最优朝向
      q_ref (N,T) 现有参考质量 · oracle (N,T) 项目自报的参考质量（用于交叉校验）
      omega_max (scalar) · window_utc (list[str])
    """
    cache = EXP / f'geometry_{site.name}_n{trials}.npz'
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    from tle_swarm_sim.simulation.experiment import _load_records
    from tle_swarm_sim.simulation.scenario_c import (
        build_aperture_task, group_records_for_window, select_aperture_window,
    )

    cfg = build_cfg(site)
    records = _load_records(cfg)

    phi_l, w_l, valid_l, thr_l, thd_l, qref_l, orc_l, win_l = [], [], [], [], [], [], [], []
    omega = None
    from refgenerator import angle_of, quality_on_grid
    for trial in range(trials):
        win = select_aperture_window(records, cfg, trial)
        grp = group_records_for_window(records, win)
        _task, geo = build_aperture_task(grp, datetime.fromisoformat(win['start_utc']), cfg)

        u_hat = np.asarray(geo['u_hat'], dtype=float)
        prio = np.asarray(geo['priority_w'], dtype=float)
        valid = np.asarray(geo['valid'], dtype=bool)
        v_ref = np.asarray(geo['v_star'], dtype=float)
        v_dem = np.asarray(geo['v_demand'], dtype=float)
        omega = float(geo['omega_max_rad_s'])

        phi = np.arctan2(u_hat[..., 1], u_hat[..., 0])
        th_ref = angle_of(v_ref)
        th_dem = angle_of(v_dem)
        grid = np.arange(1800) * (np.pi / 1800)
        q_ref = np.full(len(valid), np.nan)
        for i in np.flatnonzero(valid):
            q_ref[i] = quality_on_grid(np.array([th_ref[i]]), phi[i], prio[i], L_REF, SIGMA)[0]

        phi_l.append(phi); w_l.append(prio); valid_l.append(valid)
        thr_l.append(th_ref); thd_l.append(th_dem); qref_l.append(q_ref)
        orc_l.append(np.asarray(geo['oracle_gain_step'], dtype=float))
        win_l.append(win['start_utc'])

    out = dict(
        phi=np.stack(phi_l), w=np.stack(w_l), valid=np.stack(valid_l),
        th_ref=np.stack(thr_l), th_dem=np.stack(thd_l),
        q_ref=np.stack(qref_l), oracle=np.stack(orc_l),
        omega_max=np.array([omega]), window_utc=np.array(win_l),
        site=np.array([site.name]), n_grid_eval=np.array([1800]),
    )
    EXP.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
