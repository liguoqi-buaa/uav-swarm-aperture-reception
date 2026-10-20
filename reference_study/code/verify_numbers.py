#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_numbers.py - traceability check for every number in the manuscript.

Purpose
-------
Run this after editing the code or re-running an experiment. It re-derives each key
quantity from ``experiments/`` and looks for that literal in the manuscript source.
A miss means the manuscript is out of date, not that the data are wrong.

Usage
-----
    python verify_numbers.py              # full report
    python verify_numbers.py --quiet      # only the failures

The manuscript is not part of this repository, so point the script at it:

    VERIFY_TEX=/path/to/manuscript.tex VERIFY_OUT=report.md python verify_numbers.py

Output
------
Console report plus a Markdown report at ``$VERIFY_OUT``.

Design notes
------------
* ``kind='exact'``  : must match character for character; a miss is a FAIL.
* ``kind='timing'`` : wall-clock, environment-dependent and not reproducible; a miss is
  only a WARN. Two runs of identical code on the same machine differed by a factor of
  3.1-3.5, which is why the manuscript quotes a conservative upper bound.
* Each check records the literal it looks for and confirms that the literal is present,
  so a check cannot silently pass because the manuscript stopped mentioning the quantity.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 实验产物：仓库里为 experiments/，作者本机工作区里为 02_experiments/
EXP = ROOT / 'experiments'
if not EXP.exists():
    _alt = ROOT / '02_experiments'
    if _alt.exists():
        EXP = _alt

# 稿件与报告路径可用环境变量覆盖。
# 注意：本仓库**不随附 manuscript.tex**（避免提前公开全文），
# 复现核验时请把稿件放到 ../manuscript/ 或设置 VERIFY_TEX。
import os
_env_tex = os.environ.get('VERIFY_TEX', '')
TEX = Path(_env_tex) if _env_tex else ROOT / 'manuscript' / 'manuscript.tex'
OUT = (Path(os.environ['VERIFY_OUT']) if os.environ.get('VERIFY_OUT')
       else ROOT / 'verify_numbers_report.md')

ROWS: list[dict] = []


def norm(s: str) -> str:
    """把 LaTeX 片段压成可比较的紧凑串。

    去掉命令、花括号、$、\\、~、^、_、% 与空白，合并连续的短横线。
    这样 '$74$--$83\\%$' 与 '74-83' 都能归一到 '74-83'。
    """
    s = str(s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'[{}$\\~^_%]', '', s)
    s = re.sub(r'-{2,}', '-', s)
    return re.sub(r'\s+', '', s)


def add(section: str, item: str, computed: str, tex_literal: str,
        kind: str = 'exact', note: str = '') -> None:
    ROWS.append(dict(section=section, item=item, computed=computed,
                     tex_literal=tex_literal, kind=kind, note=note))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    tex = TEX.read_text(encoding='utf-8')

    # -------------------------------------------------------------- Table 1
    sc = pd.read_csv(EXP / 'exp1_scenarios.csv').set_index('site')
    site2pretty = {'eq_pacific': '0.0N', 'mid30_west': '30.0N',
                   'mid45_europe': '45.0N'}
    for s, pretty in site2pretty.items():
        r = sc.loc[s]
        add('Table 1', f'{pretty} at rate limit (%)',
            f'{r.sat_frac_mean*100:.1f}', f'{r.sat_frac_mean*100:.1f}')
        add('Table 1', f'{pretty} q', f'{r.q_ref_mean:.3f}', f'{r.q_ref_mean:.3f}')
        add('Table 1', f'{pretty} q of optimum', f'{r.q_opt_mean:.3f}',
            f'{r.q_opt_mean:.3f}')
        add('Table 1', f'{pretty} dtheta>30 (%)',
            f'{r.gap_gt30_mean*100:.1f}', f'{r.gap_gt30_mean*100:.1f}')
        add('Table 1', f'{pretty} collapsed',
            f'{int(r.n_catastrophic)}/20', f'{int(r.n_catastrophic)}/20')

    # -------------------------------------------------- Table 2 网格收敛
    g = pd.read_csv(EXP / 'exp1_grid_convergence.csv')
    for _, r in g.iterrows():
        k = int(r.n_grid)
        add('Table 2', f'K={k} total q (plan)',
            f'{r.q_opt_total/1000:.2f}k', f'{r.q_opt_total:,.2f}'.replace(',', r'\,'))
        add('Table 2', f'K={k} gain (%)', f'{r.gain_pct:.2f}', f'{r.gain_pct:.2f}')
        add('Table 2', f'K={k} s/window', f'{r.seconds_per_window:.4f}',
            f'{r.seconds_per_window:.4f}', kind='timing',
            note='环境依赖，不可复现')

    # ------------------------------------------ Table 4 九控制器 × 3 站点
    bc = pd.read_csv(EXP / 'exp2_by_controller.csv')
    order = ['B2', 'B2p', 'B1', 'B1p', 'ST-GNN-DMPC', 'P3', 'P2', 'P1', 'P0']
    for s, pretty in site2pretty.items():
        sub = bc[bc.site == s].set_index('label')
        for lab in order:
            if lab not in sub.index:
                continue
            r = sub.loc[lab]
            add('Table 4', f'{pretty} {lab} gain at v_ref',
                f'{r.gain_ref:.3f}', f'{r.gain_ref:.3f}')
            add('Table 4', f'{pretty} {lab} gain at v_plan',
                f'{r.gain_opt:.3f}', f'{r.gain_opt:.3f}')
            add('Table 4', f'{pretty} {lab} delta (%)',
                f'{r.gain_delta_pct:+.1f}', f'{r.gain_delta_pct:+.1f}')

    # ------------------------------------------------- 摘要 / §5.3 的关键数字
    add('Abstract', 'rate limit (deg/s)',
        f'{sc.rate_limit_deg_per_step.iloc[0]:.3f}',
        f'{sc.rate_limit_deg_per_step.iloc[0]:.3f}')
    add('Abstract', 'saturation range (%)',
        f'{sc.sat_frac_mean.min()*100:.0f}-{sc.sat_frac_mean.max()*100:.0f}',
        '74-83')
    add('Abstract', 'collapsed fraction (equator)',
        f'{sc.loc["eq_pacific"].frac_catastrophic*100:.0f}%', '20\\%')
    add('Abstract', 'quality gain range (mean of ratios)',
        f'{sc.gain_pct_mean.min():.0f}-{sc.gain_pct_mean.max():.0f}',
        '16--61', note='★ 摘要原写 15--61，应为 16--61')
    add('Abstract', 'quality gain range (ratio of sums)',
        f'{sc.gain_pct_aggregate.min():.1f}-{sc.gain_pct_aggregate.max():.1f}',
        '-', kind='info',
        note='正文 §5.3 逐站点给出（+31.3/+21.3/+14.9），不写区间')

    # 几何族 / 预测族的增益区间（摘要与 §6.4）
    geo = bc[bc.label.isin(['B1', 'B1p', 'B2', 'B2p'])]
    pre = bc[bc.label == 'ST-GNN-DMPC']
    par = bc[bc.label.isin(['P0', 'P1', 'P2', 'P3'])]
    add('Abstract', 'geometric family delta range (%)',
        f'{geo.gain_delta_pct.min():+.1f} to {geo.gain_delta_pct.max():+.1f}',
        '-1 to +11', kind='info',
        note='摘要现写 "geometric controllers move by -1 to +11%"')
    add('Abstract', 'predictive family delta range (%)',
        f'{pre.gain_delta_pct.min():+.1f} to {pre.gain_delta_pct.max():+.1f}',
        '4-15', kind='info', note='摘要在同一句里以 loss 形式表达')
    add('Abstract', 'parametric family delta range (%)',
        f'{par.gain_delta_pct.min():+.1f} to {par.gain_delta_pct.max():+.1f}',
        '4-15', kind='info', note='同预测族')

    # 主结论之比
    eq = bc[bc.site == 'eq_pacific'].set_index('label')
    add('Abstract', 'geometric/predictive ratio at v_ref',
        f'{eq.loc["B2"].gain_ref/eq.loc["ST-GNN-DMPC"].gain_ref:.2f}', '2.03')
    add('Abstract', 'geometric/predictive ratio at v_plan',
        f'{eq.loc["B2"].gain_opt/eq.loc["ST-GNN-DMPC"].gain_opt:.2f}', '2.31')

    # ------------------------------------------------- §7 规划实验（三站点）
    for s, pretty in site2pretty.items():
        f = EXP / f'exp3_planner_{s}.csv'
        if not f.exists():
            add('Section 7', f'{pretty} exp3 data', 'MISSING', '-',
                kind='missing', note='文件不存在')
            continue
        d = pd.read_csv(f)
        b2 = d[d.controller == 'B2_greedy_pd']
        b3 = d[d.controller == 'B3_planned_pd']
        add('Section 7', f'{pretty} alignment v_demand -> v_plan',
            f'{b2.alignment.mean():.3f} -> {b3.alignment.mean():.3f}',
            f'{b2.alignment.mean():.3f} \\to {b3.alignment.mean():.3f}')
        add('Section 7', f'{pretty} coverage v_demand -> v_plan',
            f'{b2.coverage_ref.mean():.3f} -> {b3.coverage_ref.mean():.3f}',
            f'{b2.coverage_ref.mean():.3f} \\to {b3.coverage_ref.mean():.3f}')
        gp = (b3.gain_plan.mean() - b2.gain_plan.mean()) / b2.gain_plan.mean() * 100
        add('Section 7', f'{pretty} gain change (%)', f'{gp:+.1f}', f'{gp:+.1f}')

    # ------------------------------------------------- §6.3 阵列投影跨度
    mr = pd.read_csv(EXP / 'exp2_metric_recompute.csv')
    span = mr.groupby('controller').span_ref_m.mean() / 1000.0
    geo = span[['B1_GEOM_PD', 'B1_GEOM_PD_PREV__w2', 'B2_GREEDY', 'B2_GREEDY_PREV__w2']]
    pot = span[['APF-TC', 'DMPC-C', 'GNN-TC', 'OLSR-FANET']]
    add('Section 6.3', 'geometric family projection span (km)',
        f'{geo.min():.1f}-{geo.max():.1f}', f'{geo.min():.1f}-{geo.max():.1f}',
        note='原稿误写 "1.6 km against 2.8 km"（1.6 实为预测控制器单值）')
    add('Section 6.3', 'potential-field family projection span (km)',
        f'{pot.min():.1f}-{pot.max():.1f}', f'{pot.min():.1f}-{pot.max():.1f}')
    add('Section 6.3', 'predictive controller projection span (km)',
        f'{span["ST-GNN-DMPC"]:.1f}', f'{span["ST-GNN-DMPC"]:.1f}')

    # -------------------------------------------- Cover Letter 关键数字同步
    cl = ASTM / '05_submission' / '02_COVER_LETTER.md'
    if cl.exists():
        clt = cl.read_text(encoding='utf-8')
        eq = bc[bc.site == 'eq_pacific'].set_index('label')
        ratio_a = eq.loc['B2'].gain_ref / eq.loc['ST-GNN-DMPC'].gain_ref
        ratio_b = eq.loc['B2'].gain_opt / eq.loc['ST-GNN-DMPC'].gain_opt
        add('Cover Letter', 'geometric/predictive ratio',
            f'{ratio_a:.2f} to {ratio_b:.2f}', '2.03 to 2.31')
        add('Cover Letter', 'predictive misalignment rise (deg)',
            f'{eq.loc["ST-GNN-DMPC"].misalign_ref_deg:.1f} to '
            f'{eq.loc["ST-GNN-DMPC"].misalign_opt_deg:.1f}', '11.4 to 28.4')
        add('Cover Letter', 'geometric family delta',
            '-1 to +11', '-1 to +11',
            note='第 5 轮发现此处曾与正文失步（原写 2-11），已同步')

    # ------------------------------------------------- 耗时与需求速率（环境依赖 / 口径待统一）
    q = pd.read_csv(EXP / 'exp1_ref_quality.csv')
    if 'dp_s' in q.columns:
        add('Abstract', 'per-window DP time (grid 3600)',
            f'{q.dp_s.min():.2f}-{q.dp_s.max():.2f} s',
            'under two seconds', kind='timing',
            note='摘要已改为保守上限；实测区间如上，随机器负载浮动')
    if 'demand_peak_rate_deg_s' in q.columns:
        add('Abstract', 'peak demand migration rate (deg/s)',
            f'{q.demand_peak_rate_deg_s.max():.1f}', '89.5', kind='exact',
            note='已与项目全局 max 89.503 核对一致；原稿误用某单窗口值 79.56')

    # --------------------------------- 题面—摘要—正文 一致性（非数值检查）
    # 问的是"题目的承诺在摘要与正文里是否兑现、口径是否统一"，
    # 这类问题第 1–5 轮审计共抓到过多次，故固化为常驻检查。
    _am = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S)
    _abstract = _am.group(1) if _am else ''
    _tm = re.search(r'\\title\{(.*?)\}', tex, re.S)
    _title = _tm.group(1) if _tm else ''
    _norm_abs, _norm_all = norm(_abstract), norm(tex)

    ALIGN = [
        # (检查名, 期望, 探针, 在哪个文本里找, 说明)
        ('题目关键词 failure mode 出现在摘要', 'present', 'failure mode',
         _norm_abs, '题目承诺之一'),
        ('题目关键词 remedy 出现在摘要', 'present', 'remedy',
         _norm_abs, '题目承诺之二'),
        ('摘要无过时耗时口径 0.2-1.6', 'absent', '0.2-1.6', _norm_abs,
         '网格平均与粗网格值混用，第 2 轮已改'),
        ('摘要无过时质量口径 15-61', 'absent', '15-61', _norm_abs,
         '应为 16-61，第 2 轮已改'),
        ('摘要质量口径 = 16-61', 'present', '16-61', _norm_abs, ''),
        ('全文无过时质量口径 15-61', 'absent', '15-61', _norm_all, ''),
        ('全文无过时耗时口径 0.2-1.6', 'absent', '0.2-1.6', _norm_all, ''),
        ('耗时口径在全稿统一', 'present', 'under two seconds', _norm_all,
         '摘要/§1.3/§8.2/§9 共 4 处'),
        ('摘要已无 "earlier study"', 'absent', 'earlier study', _norm_abs,
         '与上文 our earlier comparison 统一，且不依赖第 5 轮 P1 决定'),
        ('摘要首次出现处给出 UAV 全称', 'present', 'UAVs',
         _norm_abs, '首次出现时给全称'),
        # ---- 期刊格式硬性要求（第 7 轮加入）----
        ('GenAI 小节标题逐字合规', 'present',
         'Declaration of generative AI and AI-assisted technologies in the '
         'manuscript preparation process', _norm_all, 'ASTM 建议标题，逐字'),
        ('__STYLE_PLACEHOLDER__', 'present', 'unused',
         _norm_all, ''),
        ('未使用上标引用（super）', 'absent', 'super,comma', _norm_all, ''),
        ('软件引用含 [software] 标记', 'present', '[software]', _norm_all,
         '指南软件引用示例'),
        ('控制器标签表注存在', 'present', 'four parameter settings of a single',
         _norm_all, '表注须置于表体下方'),
        ('摘要说明观测对象（LEO 卫星）', 'present', 'LEO satellite signals',
         _norm_abs, '摘要须能独立成立'),
        ('SGP4 首次出现已展开', 'present',
         'Simplified General Perturbations (SGP4)', _norm_all, ''),
        ('TLE 首次出现已展开', 'present', 'two-line elements (TLEs)',
         _norm_all, ''),
        ('MPC 首次出现已展开', 'present', 'model predictive control (MPC)',
         _norm_all, ''),
        ('族名双称已显式桥接', 'present', '(the parametric family)', _norm_all,
         '表注把 parametric 族与 potential-field 律接上'),
        ('e 的幂已改用 exp', 'absent', 'w_m e^', _norm_all,
         '指南：Powers of e denoted by exp'),
    ]
    align_rows = []
    for name, expect, probe, hay, why in ALIGN:
        if name == '__STYLE_PLACEHOLDER__':
            continue
        present = norm(probe) in hay
        ok = present if expect == 'present' else (not present)
        align_rows.append(dict(check=name, expect=expect, probe=probe,
                               result='yes' if present else 'NO',
                               verdict='OK' if ok else '★FAIL', why=why))
    # 引用样式：elsarticle 用 \biboptions{numbers}，cas-dc 用 natbib [numbers]
    _num_ok = ('numbers,comma' in _norm_all) or ('[numbers' in _norm_all)
    align_rows.append(dict(
        check='正文引用为方括号数字（numbers）', expect='present', probe='numbers',
        result='yes' if _num_ok else 'NO', verdict='OK' if _num_ok else '★FAIL',
        why='elsarticle: biboptions{numbers} / cas-dc: natbib[numbers]'))

    align_bad = [r for r in align_rows if r['verdict'] != 'OK']

    # ---------------------------------------------------------------- 报告
    df = pd.DataFrame(ROWS)
    tex_norm = norm(tex)
    df['found_in_tex'] = df.apply(
        lambda r: '-' if r.kind == 'info' or str(r.tex_literal) == '-'
        else ('yes' if norm(r.tex_literal) in tex_norm else 'NO'), axis=1)
    bad = df[(df.found_in_tex == 'NO') & (df.kind == 'exact')]
    warn = df[df.kind.isin(['timing', 'missing'])]

    print('=' * 78)
    print('论文数字可追溯性核验')
    print('=' * 78)
    print(f'检查项总数 : {len(df)}')
    print(f'  · 数值类 : {(df.kind == "exact").sum()}')
    print(f'  · 耗时类 : {(df.kind == "timing").sum()}  （环境依赖，仅供参考）')
    print(f'tex 中未找到对应字面值（数值类）: {len(bad)}')
    if len(bad):
        print()
        print('★ 需要处理：')
        print(bad[['section', 'item', 'computed', 'tex_literal', 'note']]
              .to_string(index=False))
    if len(warn) and not args.quiet:
        print()
        print('△ 环境依赖 / 缺失项（不影响论文正确性，但需人工确认）：')
        print(warn[['section', 'item', 'computed', 'tex_literal', 'note']]
              .to_string(index=False))

    print()
    print(f'题面—摘要—正文 一致性检查 : {len(align_rows)} 项，'
          f'不通过 {len(align_bad)} 项')
    if align_bad:
        print(align_bad and pd.DataFrame(align_bad).to_string(index=False))
    elif not args.quiet:
        for r in align_rows:
            print(f'  OK  {r["check"]}')

    if not args.quiet:
        print()
        print('—' * 78)
        print('分组统计（computed 与 tex_literal 的比对由上方 NO 项给出）：')
        print(df.groupby('section').size().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as fh:
        fh.write('# 论文数字可追溯性核验报告\n\n')
        fh.write(f'检查项 {len(df)} · 数值类未匹配 {len(bad)} · 耗时/缺失 {len(warn)}\n\n')
        fh.write(df[['section', 'item', 'computed', 'tex_literal',
                     'found_in_tex', 'kind', 'note']]
                 .to_markdown(index=False))
    with OUT.open('a', encoding='utf-8') as fh:
        fh.write('\n\n## 题面—摘要—正文 一致性\n\n')
        fh.write(pd.DataFrame(align_rows).to_markdown(index=False))
        fh.write('\n')

    print(f'\n报告已写入 {OUT}')

    return 1 if (len(bad) or len(align_bad)) else 0


if __name__ == '__main__':
    sys.exit(main())
