"""Verify and publish a complete campaign's real data, figures and review table."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#012855", "#009E73", "#E69F00")
LABELS = ("BLS-LM", "SRIF", "SR-UKF")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate(source):
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] or summary["pilot"] or summary["timed_out"]:
        raise ValueError("Publish only a complete, non-pilot campaign")
    for name, expected in summary["artifact_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Campaign artifact changed: {name}")
    for name, expected in summary["source"]["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Numerical producer changed since campaign: {name}")
    rows = json.loads((source / "trials.json").read_text(encoding="utf-8"))
    plan = summary["plan"]
    expected = {(seed, scale, label) for seed in plan["seeds"] for scale in plan["noise_scales"] for label in LABELS}
    actual = {(r["seed"], r["noise_scale"], r["estimator"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicated paired trial")
    for seed in plan["seeds"]:
        for scale in plan["noise_scales"]:
            inputs = {r["input_sha256"] for r in rows if r["seed"] == seed and r["noise_scale"] == scale}
            if len(inputs) != 1:
                raise ValueError("Estimator input identities disagree")
    output = ROOT / "docs/figures/benchmark"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "trials.json", "trials.csv"):
        shutil.copy2(source / name, output / name)
    with zipfile.ZipFile(output / "raw-trials.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.glob("*.npz")):
            archive.write(path, path.name)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for field, title, unit, filename, logarithmic in (
            ("position_error_m", "Final-epoch position error", "m", "position-distribution", False),
            ("runtime_s", "Fit, endpoint projection and diagnostics runtime", "s", "runtime-distribution", True)):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
        for axis, scale in zip(axes, plan["noise_scales"]):
            values = [[r[field] for r in rows if r["noise_scale"] == scale and r["estimator"] == label and field in r]
                      for label in LABELS]
            box = axis.boxplot(values, tick_labels=LABELS, patch_artist=True, showmeans=True)
            for patch, color in zip(box["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(.45)
            axis.set(title=f"{scale:g}× noise · {len(plan['seeds'])} paired seeds", ylabel=f"{title} ({unit})")
            if logarithmic:
                axis.set_yscale("log")
        fig.suptitle("Same-model synthetic arc · same explicit prior · no operational claim")
        fig.savefig(output / (filename + ".svg"), bbox_inches="tight")
        fig.savefig(output / (filename + ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    groups = summary["analysis"]["groups"]
    lines = ["# Paired BLS-LM / SRIF / SR-UKF comparison", "",
        "This bounded Level-0 experiment uses **32 paired seeds per noise block**, "
        "64 seeded arcs and 192 estimator fits. The two noise blocks share seeds; "
        "they are not 64 independent geometries. "
        "[Predeclared design](paired-benchmark-plan.md).", "",
        f"Producing commit: `{summary['source']['git_commit']}`; "
        f"clean source: `{not summary['source']['git_dirty']}`. "
        f"Campaign elapsed time: {summary['elapsed_s']:.2f} s on the recorded environment.", "",
        "![Across-seed final position error](../figures/benchmark/position-distribution.png)", "",
        "| Noise | Estimator | Median / p95 position (m) | Median velocity (m/s) | Numerical criteria | Solver termination | Mean NEES |",
        "|---|---|---|---|---|---|---|"]
    for scale in plan["noise_scales"]:
        for label in LABELS:
            group = groups[f"noise={scale:g}/{label}"]
            pos, vel = group["position_error_m"], group["velocity_error_mps"]
            lines.append(f"| {scale:g}× | {label} | {pos['median']:.3f} / {pos['p95']:.3f} | "
                f"{vel['median']:.5f} | {group['successes']}/{group['attempted']} | "
                f"{group['solver_termination_fraction']:.1%} | {group['nees']['mean']:.3f} |")
    lines += ["", "## Interpretation and scientific review", "",
        "**Accept with caveats** for this declared synthetic comparison. All three "
        "estimators meet the predeclared numerical error/covariance criteria in all "
        "attempted trials. Solver termination is a separate criterion: SRIF reaches "
        "MaxIter in many trials even when its state error is small. Those trials "
        "are retained and are not called converged. See raw stop reasons below.", "",
        "Final errors are close in this fixed, well-behaved geometry; the data do "
        "not establish general estimator superiority. BLS-LM/SRIF estimate an initial "
        "state using the complete arc; the UKF filters sequentially. Only their "
        "aligned final epoch is compared. Priors, R, zero Q and sigma-point settings "
        "are explicit in the design and inputs.", "",
        "Final NEES has 6 degrees of freedom. For N=32 independent trials **within "
        f"one block**, the predeclared 95% mean-NEES interval is "
        f"[{groups['noise=1/BLS-LM']['mean_nees_95_interval'][0]:.3f}, "
        f"{groups['noise=1/BLS-LM']['mean_nees_95_interval'][1]:.3f}]. "
        "This is a diagnostic under the matched Gaussian prior and same-model "
        "assumptions, not a blanket calibration certificate. NIS is recorded for "
        "UKF innovations; batch post-fit residuals are stored separately. "
        "No independent-sample chi-square claim is made over the 124 correlated "
        "within-arc updates. Per-channel means, RMS and acquisition-order lag-1 "
        "correlations are in trials.json; no formal whiteness or station-geometry "
        "robustness claim is established.", "",
        "![Across-seed runtime](../figures/benchmark/runtime-distribution.png)", "",
        "Runtime includes fit, endpoint propagation and diagnostics; batch and "
        "sequential paths perform different work. Measurements are serial wall-clock "
        "values from the saved environment, without warm-up or hardware normalization. "
        "They do not establish portable speed ratios.", "",
        "## Reproduction and evidence", "",
        "```text", "python examples/benchmark_estimators.py --output outputs/benchmark", "```", "",
        "- [Summary, exact configuration, source/environment identities and all distributions](../figures/benchmark/summary.json)",
        "- [Per-trial metrics, solver stops and failure fields](../figures/benchmark/trials.csv)",
        "- [Residual/innovation channel diagnostics](../figures/benchmark/trials.json)",
        "- [Raw paired observations, truth, initial states, priors, final covariances and UKF histories](../figures/benchmark/raw-trials.zip)",
        "- [Published artifact identities](../figures/benchmark/published.json)", "",
        "The summary hashes each original NPZ member inside raw-trials.zip; "
        "published.json additionally hashes the archive and curated figures. "
        "No errors, exceptions, stopped solves or outliers are silently excluded. "
        "This same-model, idealized visibility experiment does not qualify SPICE, "
        "real station availability, model mismatch or flight performance.", ""]
    report = ROOT / "docs/research/paired-benchmark.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    receipt = {"source_summary_sha256": sha(source / "summary.json"),
               "producing_commit": summary["source"]["git_commit"],
               "artifact_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "published.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Published {len(rows)} fits with verified paired inputs and raw-data identities")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    curate(parser.parse_args().source)
