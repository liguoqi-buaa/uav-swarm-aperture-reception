from __future__ import annotations

import json
from pathlib import Path

import math
import matplotlib.pyplot as plt
import pandas as pd


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def _plot_metric(df: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(9, 4))
    plt.bar(df["algorithm"], df["mean"], yerr=df["std"], capsize=4)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _records_by_algorithm(summary: dict, section: str, metric: str) -> dict[str, dict]:
    return {row["algorithm"]: row for row in summary[section][metric]}


def _fmt(row: dict | None, decimals: int) -> str:
    if row is None:
        return "-"
    mean = row.get("mean")
    std = row.get("std")
    if mean is None or std is None or math.isnan(mean) or math.isnan(std):
        return "-"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _write_combined_table(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    df = pd.DataFrame(rows)
    path.write_text(df.to_markdown(index=False), encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")


def _write_paper_tables(summary: dict, reports_dir: Path) -> None:
    algorithms = ["OLSR-FANET", "APF-TC", "GNN-TC", "DMPC-C", "ST-GNN-DMPC"]

    def monitoring_table(section: str, title: str, path_name: str) -> None:
        perf = _records_by_algorithm(summary, section, "monitoring_performance")
        ratio = _records_by_algorithm(summary, section, "effective_monitoring_ratio")
        err = _records_by_algorithm(summary, section, "tracking_error")
        rows = [
            {
                "Algorithm": alg,
                "Average Monitoring Performance": _fmt(perf.get(alg), 2),
                "Effective Monitoring Duration Ratio (%)": _fmt(ratio.get(alg), 1),
                "Average Tracking Error (m)": _fmt(err.get(alg), 0),
            }
            for alg in algorithms
        ]
        reports_dir.joinpath(path_name).write_text(f"# {title}\n\n", encoding="utf-8")
        with reports_dir.joinpath(path_name).open("a", encoding="utf-8") as handle:
            handle.write(pd.DataFrame(rows).to_markdown(index=False))
            handle.write("\n")
        pd.DataFrame(rows).to_csv(reports_dir / path_name.replace(".md", ".csv"), index=False, encoding="utf-8")

    monitoring_table(
        "single_monitoring",
        "Table 2 Uncalibrated Monitoring Performance Comparison in Single-Satellite Scenario",
        "paper_table_2_single_monitoring_uncalibrated.md",
    )
    monitoring_table(
        "multi3_monitoring",
        "Table 3 Uncalibrated Monitoring Performance Comparison in 3-Satellite Concurrent Scenario",
        "paper_table_3_multi3_monitoring_uncalibrated.md",
    )

    conn = _records_by_algorithm(summary, "single_network", "average_connectivity")
    graph_ratio = _records_by_algorithm(summary, "single_network", "connected_graph_ratio")
    link_availability = _records_by_algorithm(summary, "single_network", "link_availability_ratio")
    k_ratio = _records_by_algorithm(summary, "single_network", "k_connectivity_satisfaction_ratio")
    disruption = _records_by_algorithm(summary, "single_network", "link_disruption_rate")
    degree = _records_by_algorithm(summary, "single_network", "average_node_degree")
    rows = [
        {
            "Algorithm": alg,
            "Average Network Connectivity": _fmt(conn.get(alg), 2),
            "Connected Graph Ratio (%)": _fmt(graph_ratio.get(alg), 1),
            "Link Availability Ratio (%)": _fmt(link_availability.get(alg), 1),
            "k-Connectivity Satisfaction Ratio (%)": _fmt(k_ratio.get(alg), 1),
            "Link Disruption Rate (%)": _fmt(disruption.get(alg), 1),
            "Average Node Degree": _fmt(degree.get(alg), 1),
        }
        for alg in algorithms
    ]
    _write_combined_table(rows, reports_dir / "paper_table_4_network_uncalibrated.md")
    _write_combined_table(rows, reports_dir / "paper_table_4_network_recomputed.md")
    if "multi3_network" in summary:
        conn3 = _records_by_algorithm(summary, "multi3_network", "average_connectivity")
        disruption3 = _records_by_algorithm(summary, "multi3_network", "link_disruption_rate")
        degree3 = _records_by_algorithm(summary, "multi3_network", "average_node_degree")
        rows3 = [
            {
                "Algorithm": alg,
                "Average Network Connectivity": _fmt(conn3.get(alg), 2),
                "Link Disruption Rate (%)": _fmt(disruption3.get(alg), 1),
                "Average Node Degree": _fmt(degree3.get(alg), 1),
            }
            for alg in algorithms
        ]
        _write_combined_table(rows3, reports_dir / "paper_table_4_network_M3_uncalibrated.md")

    conv = _records_by_algorithm(summary, "realtime", "topology_convergence_latency_ms")
    comp = _records_by_algorithm(summary, "realtime", "average_single_step_ms")
    inf = _records_by_algorithm(summary, "realtime", "inference_latency_ms")
    rows = [
        {
            "Algorithm": alg,
            "Topology Convergence Latency (ms)": _fmt(conv.get(alg), 0),
            "Average Single-Step Computational Time (ms)": _fmt(comp.get(alg), 0),
            "ST-GNN/GNN Inference Latency (ms)": _fmt(inf.get(alg), 0),
        }
        for alg in algorithms
    ]
    _write_combined_table(rows, reports_dir / "paper_table_5_realtime_uncalibrated.md")

    def extension_table(section: str, path_name: str) -> None:
        if section not in summary:
            return
        perf = _records_by_algorithm(summary, section, "monitoring_performance")
        ratio = _records_by_algorithm(summary, section, "effective_monitoring_ratio")
        err = _records_by_algorithm(summary, section, "tracking_error")
        rows = [
            {
                "Algorithm": alg,
                "Average Monitoring Performance": _fmt(perf.get(alg), 4),
                "Effective Monitoring Duration Ratio (%)": _fmt(ratio.get(alg), 2),
                "Average Tracking Error (m)": _fmt(err.get(alg), 2),
            }
            for alg in algorithms
        ]
        _write_combined_table(rows, reports_dir / path_name)

    extension_table("multi2_monitoring", "paper_table_multi2_monitoring_uncalibrated.md")
    extension_table("multi5_monitoring", "paper_table_multi5_monitoring_uncalibrated.md")

    aliases = {
        "paper_table_2_single_monitoring_uncalibrated.md": "paper_table_2_single_monitoring.md",
        "paper_table_3_multi3_monitoring_uncalibrated.md": "paper_table_3_multi3_monitoring.md",
        "paper_table_4_network_uncalibrated.md": "paper_table_4_network.md",
        "paper_table_5_realtime_uncalibrated.md": "paper_table_5_realtime.md",
    }
    for src_name, dst_name in aliases.items():
        src = reports_dir / src_name
        if src.exists():
            (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def generate_reports(summary: dict, config) -> None:
    reports_dir = config.reports_dir
    figures_dir = config.figures_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_paper_tables(summary, reports_dir)

    for metric, records in summary["single_monitoring"].items():
        df = pd.DataFrame(records)
        _write_table(df, reports_dir / f"table_single_{metric}.md")
        _plot_metric(df, f"Single-Satellite {metric}", metric, figures_dir / f"single_{metric}.png")

    for section, records_by_metric in summary.items():
        if not section.startswith("multi") or not section.endswith("_monitoring"):
            continue
        group_label = section.removeprefix("multi").removesuffix("_monitoring")
        file_prefix = section.removesuffix("_monitoring")
        for metric, records in records_by_metric.items():
            df = pd.DataFrame(records)
            _write_table(df, reports_dir / f"table_{file_prefix}_{metric}.md")
            _plot_metric(df, f"{group_label}-Satellite {metric}", metric, figures_dir / f"{file_prefix}_{metric}.png")

    for metric, records in summary["single_network"].items():
        df = pd.DataFrame(records)
        _write_table(df, reports_dir / f"table_network_{metric}.md")
        _plot_metric(df, f"Single-Satellite Network {metric}", metric, figures_dir / f"network_{metric}.png")

    for metric, records in summary["realtime"].items():
        df = pd.DataFrame(records)
        _write_table(df, reports_dir / f"table_realtime_{metric}.md")
        _plot_metric(df, f"Realtime {metric}", metric, figures_dir / f"realtime_{metric}.png")
