# 原始 TLE 输入（随论文开源）

## 文件

| 文件 | 记录数 | 说明 |
|:--|--:|:--|
| `starlink.tle` | 10,304 | 单次快照的 CelesTrak Starlink GP/TLE 查询结果，每颗卫星一条记录 |

## 数据来源与历元口径（**请勿误写为"2024–2025 历史归档"**）

- **来源**：CelesTrak `https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle`（公开 GP 数据）。
- **获取方式**：一次性快照查询，**不是**按月/按年爬取的历史归档；每颗卫星只保留一条 TLE。
- **实际历元范围**：**2026-04-29 → 2026-05-02**（不同卫星的 TLE 历元不同，跨度约 3 天，取决于 CelesTrak 各星最后一次上注时间）。
- **记录条数**：10,304 条，对应 10,304 颗不同的 Starlink 卫星（`catalog_number` 无重复），**不是**时序序列。

因此本论文实际使用的是一个 **2026 年 5 月 1 日前后、约 3 天历元跨度的单快照公开 TLE 集**，而**不是** 2024–2025 年的历史归档。论文正文与任何说明材料都应使用这一口径。

## 由原始快照派生的文件（在 `data/processed/`）

| 文件 | 记录数 | 生成方式 |
|:--|--:|:--|
| `starlink_archive.tle` | 1,329 | 在 `starlink.tle` 上施加主壳层筛选：轨道倾角 $52.5^\circ$–$53.5^\circ$、平均轨道高度 520–580 km（见 `tle_swarm_sim/config.py`），每星一条 |

论文的共视窗口池（`analysis/covisible_pool.json`，40 个三共视窗口）由 `starlink_archive.tle` 经 SGP4 传播 + 站点仰角几何计算得到，历元覆盖 2026-04-30 → 2026-05-01。

## 复现命令

```bash
# 使用仓库内已有的原始快照（推荐，无需联网）
python -m tle_swarm_sim.cli data --source-tle data/raw/starlink.tle --no-download --allow-current-epochs

# 或重新从 CelesTrak 拉取最新快照（注意：历元会与论文所用快照不同）
python -m tle_swarm_sim.cli data --allow-current-epochs
```

`--allow-current-epochs` 是必需的：`tle_swarm_sim/config.py` 的默认年筛为 `start_year=2024, end_year=2025`（模板默认值），对本仓库的 2026 历元数据会过滤为空集。若要保持与论文完全一致，请直接使用仓库内的 `data/raw/starlink.tle` 快照，不要重新下载。

## 许可与引用

TLE/GP 数据由 CelesTrak 公开发布，使用请遵循 CelesTrak 的使用条款并致谢其数据来源（T.S. Kelso, CelesTrak, https://celestrak.org）。轨道传播使用 `sgp4` 库（Vallado 等，2006）。
