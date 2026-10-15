"""只读脚本：在「等基线长度」下重算孔径对齐质量 A，剔除长度混淆。

背景
----
论文 Table 2 的 "phase alignment" 定义为
    A(p) = sum_m w_m * exp( -Var_i(p_i . u_hat_m) / sigma^2 )
其中 Var_i(p_i . u_hat_m) 对长度 L 的线阵约为 (L^2/12)*(v . u_hat_m)^2，
即 **方差正比于 L^2**：阵列越短，A 越高（与朝向无关的长度效应）。
ST-GNN-DMPC 的 span 比基线短 36.6%，因此其 "+114.9% alignment" 含长度混淆。

本脚本从原始逐 trial 轨迹**独立复算**三种量：
  1. A_raw   —— 复现论文口径（用于校验重建是否正确）
  2. A@Lref  —— 把每步 UAV 构型按 PCA 主轴等比缩放到公共基线长度 Lref 后再算 A
                （剔除长度效应，保留真实朝向与形状）
  3. A_dir   —— 纯朝向指标：理想直线阵（长度 Lref）按实际主轴朝向摆放时的 A
                （只反映朝向，与任何实测长度无关）

只读：不写 scenario_c_matched 下任何文件。

用法
----
    python normalize_alignment.py [--root <scenario_C 目录>] [--mode matched]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALGOS = ["OLSR-FANET", "APF-TC", "GNN-TC", "DMPC-C", "ST-GNN-DMPC"]
SIGMA = 500.0                 # config.aperture_sigma_m
L_NOMINAL = 3000.0            # config.aperture_baseline_m
L_REF_CANDIDATES = [3000.0, 2502.0]   # 标称基线 / 最好基线的实测 span
MIN_ELEV = 15.0               # config.aperture_min_elevation_deg


def load_trial(trial_dir: Path) -> dict:
    uav = pd.read_csv(trial_dir / "trajectory_uav.csv")
    sat = pd.read_csv(trial_dir / "trajectory_satellite.csv")
    ts = pd.read_csv(trial_dir / "metrics_timeseries.csv")
    return {"uav": uav, "sat": sat, "ts": ts}


def reconstruct_geometry(uav: pd.DataFrame, sat: pd.DataFrame):
    """从轨迹 CSV 重建 u_hat(step, M) 与 priority_w(step, M)。向量化实现。"""
    steps = np.sort(uav["step"].unique())
    m_count = sat["satellite_group_index"].nunique()

    s = sat.sort_values(["step", "satellite_group_index"])
    xy = s[["x_m", "y_m"]].to_numpy(dtype=float).reshape(len(steps), m_count, 2)
    elev = s["elevation_deg"].to_numpy(dtype=float).reshape(len(steps), m_count)
    norm = np.linalg.norm(xy, axis=2, keepdims=True)
    u_hat = xy / np.maximum(norm, 1e-9)

    w_raw = np.clip(elev - MIN_ELEV, 0.0, None)
    wsum = w_raw.sum(axis=1, keepdims=True)
    valid = wsum[:, 0] > 1e-9
    w = np.where(wsum > 1e-9, w_raw / np.maximum(wsum, 1e-9), 0.0)
    return steps, u_hat, w, valid


def positions_by_step(uav: pd.DataFrame, steps):
    """返回 (steps, N, 2) 的 xy 位置。向量化实现。"""
    n = uav["uav_id"].nunique()
    u = uav.sort_values(["step", "uav_id"])
    return u[["x_m", "y_m"]].to_numpy(dtype=float).reshape(len(steps), n, 2)


def alignment_of(xy: np.ndarray, u_hat: np.ndarray, w: np.ndarray, sigma: float) -> float:
    """A = sum_m w_m exp(-Var_i(p_i . u_hat_m)/sigma^2)，单步。"""
    s = xy @ u_hat.T                     # (N, M)
    var_m = s.var(axis=0)                # (M,)
    return float(np.sum(w * np.exp(-var_m / max(sigma ** 2, 1e-9))))


def major_span(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA 主轴单位向量与该轴上的跨度。"""
    centered = xy - xy.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(xy.shape[0], 1)
    try:
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
    except np.linalg.LinAlgError:
        major = np.array([1.0, 0.0])
    r = centered @ major
    return major, float(r.max() - r.min())


def per_trial_metrics(trial_dir: Path) -> dict:
    data = load_trial(trial_dir)
    steps, u_hat, w, valid = reconstruct_geometry(data["uav"], data["sat"])
    pos = positions_by_step(data["uav"], steps)

    n_steps = len(steps)
    a_raw_all = np.full(n_steps, np.nan)
    a_raw_valid = np.full(n_steps, np.nan)
    a_at = {f"A@{int(L)}": np.full(n_steps, np.nan) for L in L_REF_CANDIDATES}
    a_dir = {f"Adir@{int(L)}": np.full(n_steps, np.nan) for L in L_REF_CANDIDATES}

    for i in range(n_steps):
        xy = pos[i]
        a_raw_all[i] = alignment_of(xy, u_hat[i], w[i], SIGMA)
        if valid[i]:
            a_raw_valid[i] = a_raw_all[i]

        major, span = major_span(xy)
        for L in L_REF_CANDIDATES:
            if span > 1e-6:
                k = L / span
                # 等比缩放：var 乘以 k^2
                s = xy @ u_hat[i].T
                var = s.var(axis=0) * k * k
                a_at[f"A@{int(L)}"][i] = float(np.sum(w[i] * np.exp(-var / SIGMA ** 2)))
            # 纯朝向：理想直线阵长度 L，沿实际主轴 major
            cos2 = (u_hat[i] @ major) ** 2
            var_ideal = (L ** 2 / 12.0) * cos2
            a_dir[f"Adir@{int(L)}"][i] = float(np.sum(w[i] * np.exp(-var_ideal / SIGMA ** 2)))

    key = "aperture_alignment"
    csv_mean = float(ts_mean(data["ts"], key))
    out = {
        "trial": trial_dir.name,
        "A_csv": csv_mean,
        "A_raw_all": float(np.nanmean(a_raw_all)),
        "A_raw_valid": float(np.nanmean(a_raw_valid)) if np.isfinite(a_raw_valid).any() else np.nan,
        "valid_fraction": float(valid.mean()),
    }
    for k, arr in {**a_at, **a_dir}.items():
        out[k] = float(np.nanmean(arr))
    return out


def ts_mean(ts: pd.DataFrame, key: str) -> float:
    v = pd.to_numeric(ts[key], errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def main() -> int:
    global MIN_ELEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "uav_swarm_scenario_C"))
    ap.add_argument("--mode", default="matched")
    ap.add_argument("--min-elev", type=float, default=None,
                    help="权重参考仰角阈值；用于判定原始 trial 实际使用的阈值")
    ap.add_argument("--reports-dir", type=str, default=None,
                    help="直接指定 reports 目录（含 trials/）；默认 <root>/reports/scenario_c_<mode>")
    ap.add_argument("--algos", type=str, default=None,
                    help="逗号分隔的算法名；默认使用论文原 5 个 profile 名")
    args = ap.parse_args()

    if args.min_elev is not None:
        MIN_ELEV = args.min_elev

    base = Path(args.reports_dir) if args.reports_dir else (
        Path(args.root) / "reports" / f"scenario_c_{args.mode}")
    trials_root = base / "trials"
    algos = [a.strip() for a in args.algos.split(",")] if args.algos else ALGOS
    if not trials_root.exists():
        print(f"[ERROR] 未找到 {trials_root}")
        return 1

    records = []
    for algo in algos:
        algo_dir = trials_root / "M3" / algo
        if not algo_dir.exists():
            print(f"[WARN] 缺目录 {algo_dir}")
            continue
        for trial_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            rec = per_trial_metrics(trial_dir)
            rec["algorithm"] = algo
            records.append(rec)
        print(f"  已处理 {algo}: {sum(1 for r in records if r['algorithm'] == algo)} trials")

    df = pd.DataFrame(records)
    out_name = f"alignment_normalized_{Path(base).name}.csv"
    df.to_csv(Path(__file__).resolve().parent / out_name, index=False)

    print()
    print("=" * 104)
    print(f"等基线长度重算结果（{base.name}, n={len(df)//max(len(algos),1)} trials/算法, sigma={SIGMA:g} m, 仰角阈值={MIN_ELEV:g}°）")
    print("=" * 104)
    cols = ["A_csv", "A_raw_all", "A_raw_valid"] + \
           [f"A@{int(L)}" for L in L_REF_CANDIDATES] + \
           [f"Adir@{int(L)}" for L in L_REF_CANDIDATES]
    header = f"{'algorithm':<14}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        row = f"{algo:<14}"
        for c in cols:
            row += f"{sub[c].mean():>9.4f}±{sub[c].std(ddof=1):<4.3f}"
        print(row)

    print()
    print("=== 校验：重建的 A_raw_all 与 CSV 的 aperture_alignment 是否一致 ===")
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        d = (sub["A_raw_all"] - sub["A_csv"]).abs()
        rel = d.mean() / sub["A_csv"].mean()
        print(f"  {algo:<14} 平均绝对偏差={d.mean():.4f}  相对={100*rel:.2f}%  最大={d.max():.4f}")

    print()
    print("=== 诊断：仰角有效步数（w_raw>0 即至少一颗星高于仰角阈值）===")
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        print(f"  {algo:<14} valid_fraction = {sub['valid_fraction'].mean():.4f} ± {sub['valid_fraction'].std(ddof=1):.4f}")

    print()
    print("=== 配对显著性检验（ST-GNN-DMPC vs 每个 profile，按 trial 索引配对）===")
    try:
        from scipy import stats as _st
    except Exception:
        _st = None
    for c in ["A_raw_all"] + [f"A@{int(L)}" for L in L_REF_CANDIDATES] + [f"Adir@{int(L)}" for L in L_REF_CANDIDATES]:
        st_vals = df[df["algorithm"] == "ST-GNN-DMPC"][c].to_numpy()
        print(f"  -- {c} --")
        for algo in algos:
            if algo == "ST-GNN-DMPC":
                continue
            b_vals = df[df["algorithm"] == algo][c].to_numpy()
            n = min(len(st_vals), len(b_vals))
            a, b = st_vals[:n], b_vals[:n]
            if _st is not None:
                t, p = _st.ttest_rel(a, b)
                w, pw = _st.wilcoxon(a, b)
                print(f"     vs {algo:<14} 提升={100*(a.mean()-b.mean())/b.mean():+7.1f}%   "
                      f"paired t p={p:.2e}   Wilcoxon p={pw:.2e}")
            else:
                print(f"     vs {algo:<14} 提升={100*(a.mean()-b.mean())/b.mean():+7.1f}%")

    print()
    print("=== 关键对比：原始 vs 等长度 vs 纯朝向 ===")
    base_algo = "DMPC-C"
    b = df[df["algorithm"] == base_algo]
    combos = [("A_raw_all", "实测 A（含长度效应）"),
              ("A@3000", "等长度 A @ L=3000 m（标称基线）"),
              ("A@2502", "等长度 A @ L=2502 m（最好基线实测 span）"),
              ("Adir@3000", "纯朝向 Adir @ L=3000 m"),
              ("Adir@2502", "纯朝向 Adir @ L=2502 m")]
    for c, name in combos:
        bm = b[c].mean()
        sm = df[df["algorithm"] == "ST-GNN-DMPC"][c].mean()
        print(f"  {name:<38} DMPC-C={bm:.4f}  ST={sm:.4f}  提升={100*(sm-bm)/bm:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
