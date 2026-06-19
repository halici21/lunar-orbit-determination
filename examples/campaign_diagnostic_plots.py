"""Build visibility/error diagnostic plots from saved campaign CSV outputs.

Run from the project root after the campaign reports have been generated:

    python python_port/examples/campaign_diagnostic_plots.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("python_port") / "results"


def main() -> None:
    out_paths = [
        plot_28day_itu_visibility_summary(RESULTS_DIR),
        plot_28day_itu_all_visibility_arcs(RESULTS_DIR),
        write_28day_itu_all_visibility_arcs_csv(RESULTS_DIR),
        plot_28day_itu_arc_diagnostics(RESULTS_DIR),
    ]
    for path in out_paths:
        print(f"Wrote {path}")


def plot_28day_itu_visibility_summary(results_dir: Path) -> Path:
    metrics, arcs = _read_visibility_summary(results_dir / "campaign_28day_itu_visibility_summary.csv")
    selected_rows = _read_selected_arc_rows(results_dir / "campaign_28day_itu_selected_arcs.csv")

    raw_hours = _visible_hours_from_samples(metrics, arcs)
    raw_fraction = metrics.get(("network_raw_visible_fraction", "network"), np.nan)
    filled_fraction = metrics.get(("network_filled_visible_fraction", "network"), np.nan)
    num_arcs = metrics.get(("num_arcs", "network"), np.nan)
    od_ready_arcs = sum(1 for arc in arcs if arc["raw_measurement_samples"] >= 4)

    durations = np.array([arc["duration_s"] / 3600.0 for arc in arcs], dtype=float)
    hot = [row for row in selected_rows if row["scenario"].lower().endswith("hot")]
    cold = [row for row in selected_rows if row["scenario"].lower().endswith("cold")]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    fig.suptitle("28-Day ITU-only ground-station visibility and OD sanity")

    ax = axes[0, 0]
    ax.axis("off")
    metric_lines = [
        f"Visible sample time: {raw_hours:.2f} h",
        f"Raw visibility fraction: {raw_fraction:.3f}",
        f"Gap-filled visibility fraction: {filled_fraction:.3f}",
        f"Visibility arcs: {int(num_arcs)}",
        f"OD-ready arcs (>=4 samples): {od_ready_arcs}",
        f"Median arc duration: {np.nanmedian(durations):.2f} h",
        f"Max arc duration: {np.nanmax(durations):.2f} h",
    ]
    ax.text(
        0.04,
        0.92,
        "\n".join(metric_lines),
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.55,
        bbox={"facecolor": "#f5f7f8", "edgecolor": "#c7d0d5", "boxstyle": "round,pad=0.5"},
    )
    ax.set_title("ITU-only visibility headline metrics")

    ax = axes[0, 1]
    ax.hist(durations, bins=_duration_bins(durations), color="#5b8e3e", edgecolor="white")
    ax.axvline(np.nanmedian(durations), color="#b83232", linestyle="--", linewidth=1.6, label="median")
    ax.set_xlabel("visibility arc duration [h]")
    ax.set_ylabel("arc count")
    ax.set_title("All ITU visibility arc durations")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for scenario_rows, color, label in [(cold, "#2f6f9f", "cold"), (hot, "#b83232", "hot")]:
        starts = np.array([row["start_h"] for row in scenario_rows], dtype=float)
        errors = np.array([row["final_position_error_m"] for row in scenario_rows], dtype=float)
        ax.semilogy(starts, np.maximum(errors, 1e-12), marker="o", color=color, label=label)
    ax.set_xlabel("selected arc start [h]")
    ax.set_ylabel("final position error [m]")
    ax.set_title("Selected ITU arc OD errors")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for scenario_rows, marker, label in [(cold, "o", "cold"), (hot, "s", "hot")]:
        duration = np.array([row["arc_span_h"] for row in scenario_rows], dtype=float)
        errors = np.array([row["final_position_error_m"] for row in scenario_rows], dtype=float)
        cond = np.array([row["condition_number"] for row in scenario_rows], dtype=float)
        sc = ax.scatter(
            duration,
            np.maximum(errors, 1e-12),
            c=np.log10(np.maximum(cond, 1.0)),
            cmap="viridis",
            marker=marker,
            s=70,
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("selected arc duration [h]")
    ax.set_ylabel("final position error [m]")
    ax.set_title("ITU duration vs error, color = log10(condition)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log10 condition number")

    path = results_dir / "campaign_28day_itu_only_summary_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_28day_itu_all_visibility_arcs(results_dir: Path) -> Path:
    _, arcs = _read_visibility_summary(results_dir / "campaign_28day_itu_visibility_summary.csv")
    selected_rows = _read_selected_arc_rows(results_dir / "campaign_28day_itu_selected_arcs.csv")
    selected_arc_ids = sorted({int(row["arc_id"]) for row in selected_rows})

    arc_ids = np.array([arc["arc_id"] for arc in arcs], dtype=int)
    start_h = np.array([arc["start_s"] / 3600.0 for arc in arcs], dtype=float)
    duration_h = np.array([arc["duration_s"] / 3600.0 for arc in arcs], dtype=float)
    samples = np.array([arc["raw_measurement_samples"] for arc in arcs], dtype=int)
    od_ready = samples >= 4
    selected = np.isin(arc_ids, selected_arc_ids)

    fig, axes = plt.subplots(4, 1, figsize=(14.0, 10.0), constrained_layout=True)
    fig.suptitle("28-Day ITU-only: all visibility arcs")

    ax = axes[0]
    for arc_id, arc_start_h, arc_duration_h, ready, chosen in zip(arc_ids, start_h, duration_h, od_ready, selected):
        color = "#b83232" if chosen else "#2f6f9f"
        alpha = 0.85 if chosen else (0.42 if ready else 0.15)
        height = 0.62 if chosen else 0.46
        ax.broken_barh([(arc_start_h, max(arc_duration_h, 1.0 / 3600.0))], (0.5 - height / 2, height), facecolors=color, alpha=alpha)
        if chosen:
            ax.text(arc_start_h, 1.02, str(arc_id), fontsize=7, rotation=90, ha="center", va="bottom", color="#7f1d1d")
    ax.set_xlim(0.0, max(start_h + duration_h) + 6.0)
    ax.set_yticks([])
    ax.set_xlabel("campaign time [h]")
    ax.set_title("All ITU visibility windows; red labels are OD-selected representative arcs")

    ax = axes[1]
    bar_colors = np.where(selected, "#b83232", np.where(od_ready, "#5b8e3e", "#b7c3cb"))
    ax.bar(arc_ids, duration_h, color=bar_colors, width=0.85)
    ax.axhline(float(np.nanmedian(duration_h)), color="#222222", linestyle="--", linewidth=1.0, label="median")
    ax.set_ylabel("duration [h]")
    ax.set_title("Arc duration by arc id")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(arc_ids, samples, color=bar_colors, width=0.85)
    ax.axhline(4, color="#b83232", linestyle="--", linewidth=1.0, label="OD-ready threshold")
    ax.set_ylabel("raw samples")
    ax.set_title("Raw visible samples by arc id")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[3]
    cumulative_visible_h = np.cumsum(samples * _sample_step_h_from_arcs(arcs))
    ax.plot(arc_ids, cumulative_visible_h, color="#2f6f9f", linewidth=1.8)
    ax.scatter(arc_ids[selected], cumulative_visible_h[selected], color="#b83232", s=26, label="OD-selected")
    ax.set_xlabel("arc id")
    ax.set_ylabel("cumulative visible sample time [h]")
    ax.set_title("Cumulative ITU visible tracking time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    path = results_dir / "campaign_28day_itu_all_visibility_arcs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_28day_itu_all_visibility_arcs_csv(results_dir: Path) -> Path:
    _, arcs = _read_visibility_summary(results_dir / "campaign_28day_itu_visibility_summary.csv")
    selected_rows = _read_selected_arc_rows(results_dir / "campaign_28day_itu_selected_arcs.csv")
    selected_arc_ids = sorted({int(row["arc_id"]) for row in selected_rows})

    path = results_dir / "campaign_28day_itu_all_visibility_arcs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arc_id",
                "start_h",
                "end_h",
                "duration_h",
                "raw_measurement_samples",
                "od_ready_min4",
                "od_selected_representative",
            ]
        )
        for arc in arcs:
            writer.writerow(
                [
                    arc["arc_id"],
                    arc["start_s"] / 3600.0,
                    arc["end_s"] / 3600.0,
                    arc["duration_s"] / 3600.0,
                    arc["raw_measurement_samples"],
                    int(arc["raw_measurement_samples"] >= 4),
                    int(arc["arc_id"] in selected_arc_ids),
                ]
            )
    return path


def plot_campaign_visibility_summary(results_dir: Path) -> Path:
    rows = []
    for summary_name, label in [
        ("campaign_4day_single_canberra_visibility_summary.csv", "4d Canberra"),
        ("campaign_4day_multi_dsn_itu_visibility_summary.csv", "4d DSN+ITU"),
        ("campaign_4day_multi_extended_visibility_summary.csv", "4d Extended"),
        ("campaign_28day_itu_visibility_summary.csv", "28d ITU"),
    ]:
        metrics, arcs = _read_visibility_summary(results_dir / summary_name)
        raw_fraction = metrics.get(("network_raw_visible_fraction", "network"), np.nan)
        raw_hours = _visible_hours_from_samples(metrics, arcs)
        rows.append(
            {
                "label": label,
                "raw_fraction": raw_fraction,
                "visible_hours": raw_hours,
                "num_arcs": metrics.get(("num_arcs", "network"), np.nan),
            }
        )

    labels = [row["label"] for row in rows]
    visible_hours = np.array([row["visible_hours"] for row in rows], dtype=float)
    fractions = np.array([row["raw_fraction"] for row in rows], dtype=float)
    arc_counts = np.array([row["num_arcs"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    x = np.arange(len(labels))

    axes[0].bar(x, visible_hours, color="#2f6f9f")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("raw visible sample time [h]")
    axes[0].set_title("Ground-station raw visibility hours")

    axes[1].bar(x, fractions, color="#5b8e3e")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("raw visibility fraction")
    axes[1].set_ylim(0.0, max(0.75, np.nanmax(fractions) * 1.15))
    axes[1].set_title("Visibility fraction")

    axes[2].bar(x, arc_counts, color="#9a5b2f")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_ylabel("arc count")
    axes[2].set_title("Visibility arc count")

    path = results_dir / "campaign_visibility_summary_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_4day_arc_diagnostics(results_dir: Path) -> Path:
    _, visibility_arcs = _read_visibility_summary(results_dir / "campaign_4day_multi_dsn_itu_visibility_summary.csv")
    od_rows = _read_od_rows(results_dir / "campaign_4day_rr_od_summary.csv")
    arc_by_id = {int(row["arc_id"]): row for row in visibility_arcs}
    merged = _merge_od_with_visibility(od_rows, arc_by_id)

    return _plot_arc_diagnostics(
        merged,
        visibility_arcs,
        results_dir / "campaign_4day_visibility_error_diagnostics.png",
        "4-Day DSN+ITU clean geometric RR",
    )


def plot_28day_itu_arc_diagnostics(results_dir: Path) -> Path:
    _, visibility_arcs = _read_visibility_summary(results_dir / "campaign_28day_itu_visibility_summary.csv")
    selected_rows = _read_selected_arc_rows(results_dir / "campaign_28day_itu_selected_arcs.csv")

    return _plot_arc_diagnostics(
        selected_rows,
        visibility_arcs,
        results_dir / "campaign_28day_itu_visibility_error_diagnostics.png",
        "28-Day ITU-only clean geometric RR",
    )


def _plot_arc_diagnostics(rows: list[dict], visibility_arcs: list[dict], output_path: Path, title: str) -> Path:
    scenarios = sorted({row["scenario"] for row in rows})
    colors = _scenario_colors(scenarios)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    for arc in visibility_arcs:
        start_h = arc["start_s"] / 3600.0
        duration_h = arc["duration_s"] / 3600.0
        alpha = 0.45 if arc["raw_measurement_samples"] >= 4 else 0.18
        ax.broken_barh([(start_h, max(duration_h, 1e-6))], (0.25, 0.5), facecolors="#2f6f9f", alpha=alpha)
    selected_arc_ids = sorted({int(row["arc_id"]) for row in rows})
    selected_starts = [
        next((arc["start_s"] / 3600.0 for arc in visibility_arcs if int(arc["arc_id"]) == arc_id), np.nan)
        for arc_id in selected_arc_ids
    ]
    ax.scatter(selected_starts, np.full(len(selected_starts), 1.05), marker="v", color="#b83232", s=22, label="OD-selected arc")
    ax.set_yticks([])
    ax.set_xlabel("campaign time [h]")
    ax.set_title("Earth station visibility windows")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    durations = np.array([arc["duration_s"] / 3600.0 for arc in visibility_arcs], dtype=float)
    bins = _duration_bins(durations)
    ax.hist(durations, bins=bins, color="#5b8e3e", edgecolor="white")
    ax.set_xlabel("visibility arc duration [h]")
    ax.set_ylabel("arc count")
    ax.set_title("Visibility duration distribution")

    ax = axes[1, 0]
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        starts = np.array([row["start_h"] for row in scenario_rows], dtype=float)
        errors = np.array([row["final_position_error_m"] for row in scenario_rows], dtype=float)
        obs = np.array([row["num_observations"] for row in scenario_rows], dtype=float)
        sizes = 25.0 + 4.0 * np.clip(obs, 0.0, 20.0)
        ax.semilogy(starts, np.maximum(errors, 1e-12), marker="o", markersize=4, color=colors[scenario], label=scenario)
        ax.scatter(starts, np.maximum(errors, 1e-12), s=sizes, color=colors[scenario], alpha=0.35)
    ax.set_xlabel("arc start [h]")
    ax.set_ylabel("final position error [m]")
    ax.set_title("OD error over campaign time")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        duration = np.array([row["arc_span_h"] for row in scenario_rows], dtype=float)
        errors = np.array([row["final_position_error_m"] for row in scenario_rows], dtype=float)
        cond = np.array([row["condition_number"] for row in scenario_rows], dtype=float)
        log_cond = np.log10(np.maximum(cond, 1.0))
        sc = ax.scatter(
            duration,
            np.maximum(errors, 1e-12),
            c=log_cond,
            cmap="viridis",
            marker=_scenario_marker(scenario),
            s=70,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.85,
            label=scenario,
        )
    ax.set_yscale("log")
    ax.set_xlabel("selected arc duration [h]")
    ax.set_ylabel("final position error [m]")
    ax.set_title("Duration is not enough: color = log10(condition)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log10 condition number")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _read_visibility_summary(path: Path) -> tuple[dict[tuple[str, str], float], list[dict]]:
    metrics: dict[tuple[str, str], float] = {}
    arcs: list[dict] = []
    section = "metrics"
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[:5] == ["arc_id", "start_s", "end_s", "duration_s", "raw_measurement_samples"]:
                section = "arcs"
                continue
            if section == "metrics":
                if row[:3] == ["metric", "name", "value"]:
                    continue
                if len(row) >= 3:
                    metrics[(row[0], row[1])] = float(row[2])
            elif len(row) >= 5:
                arcs.append(
                    {
                        "arc_id": int(row[0]),
                        "start_s": float(row[1]),
                        "end_s": float(row[2]),
                        "duration_s": float(row[3]),
                        "raw_measurement_samples": int(row[4]),
                    }
                )
    return metrics, arcs


def _read_od_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_selected_arc_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "arc_id": int(row["arc_id"]),
                    "start_h": float(row["start_h"]),
                    "end_h": float(row["end_h"]),
                    "arc_span_h": float(row["arc_span_h"]),
                    "visible_samples": int(row["visible_samples"]),
                    "scenario": row["scenario"],
                    "num_observations": int(row["num_observations"]),
                    "final_position_error_m": float(row["final_position_error_m"]),
                    "condition_number": float(row["condition_number"]),
                }
            )
        return rows


def _merge_od_with_visibility(od_rows: list[dict], arc_by_id: dict[int, dict]) -> list[dict]:
    rows = []
    for row in od_rows:
        arc_id = int(row["arc_id"])
        arc = arc_by_id[arc_id]
        rows.append(
            {
                "arc_id": arc_id,
                "start_h": arc["start_s"] / 3600.0,
                "end_h": arc["end_s"] / 3600.0,
                "arc_span_h": arc["duration_s"] / 3600.0,
                "visible_samples": arc["raw_measurement_samples"],
                "scenario": row["scenario"],
                "num_observations": int(row["num_observations"]),
                "final_position_error_m": float(row["final_position_error_m"]),
                "condition_number": float(row["condition_number"]),
            }
        )
    return rows


def _visible_hours_from_samples(metrics: dict[tuple[str, str], float], arcs: list[dict]) -> float:
    raw_samples = metrics.get(("network_raw_visible_samples", "network"), np.nan)
    if not arcs or not np.isfinite(raw_samples):
        return np.nan
    sample_steps = []
    for arc in arcs:
        samples = arc["raw_measurement_samples"]
        if samples > 1:
            sample_steps.append(arc["duration_s"] / (samples - 1))
    if not sample_steps:
        return np.nan
    sample_step_s = float(np.median(sample_steps))
    return raw_samples * sample_step_s / 3600.0


def _sample_step_h_from_arcs(arcs: list[dict]) -> float:
    sample_steps_s = []
    for arc in arcs:
        samples = arc["raw_measurement_samples"]
        if samples > 1:
            sample_steps_s.append(arc["duration_s"] / (samples - 1))
    if not sample_steps_s:
        return 0.0
    return float(np.median(sample_steps_s) / 3600.0)


def _duration_bins(durations: np.ndarray) -> np.ndarray:
    max_duration = float(np.nanmax(durations)) if durations.size else 1.0
    if max_duration <= 2.0:
        return np.arange(0.0, max_duration + 0.25, 0.25)
    return np.linspace(0.0, max_duration, 24)


def _scenario_colors(scenarios: list[str]) -> dict[str, str]:
    palette = ["#2f6f9f", "#b83232", "#5b8e3e", "#6b4fa3", "#9a5b2f"]
    return {scenario: palette[i % len(palette)] for i, scenario in enumerate(scenarios)}


def _scenario_marker(scenario: str) -> str:
    lower = scenario.lower()
    if "cold" in lower:
        return "o"
    if "hot" in lower:
        return "s"
    if "sqrt" in lower:
        return "^"
    return "D"


if __name__ == "__main__":
    main()
