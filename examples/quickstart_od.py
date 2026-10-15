"""Fit one seeded, idealized range/azimuth/elevation arc without SPICE kernels.

Run from an installed checkout: python examples/quickstart_od.py
This educational same-model example is not a realistic visibility campaign.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from lunar_od import (
    PassGeometry, Station, compute_position_residuals_analytic,
    estimate_position_bls_lm, estimate_position_srif, propagate_augmented_state,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/configs/quickstart.json"
COLORS = {"BLS-LM": "#012855", "SRIF": "#0f9b8e"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    """A Git revision plus file hashes also identifies a source-only release."""
    try:
        top = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if Path(top).resolve() != ROOT:
            raise OSError("Source archive is inside another repository")
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
        release_record = ROOT / "SOURCE_REVISION.json"
        if release_record.is_file():
            commit = json.loads(release_record.read_text())["git_commit"]
    files = [Path(__file__).resolve(), *sorted((ROOT / "src/lunar_od").rglob("*.py"))]
    return {"git_commit": commit, "git_dirty": dirty,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}}


def build_case(config):
    times = np.arange(0.0, config["duration_s"] + 0.5 * config["sample_step_s"],
                      config["sample_step_s"])
    earth = np.asarray(config["earth_position_m"], dtype=float)
    get_earth = lambda t: np.tile(earth, (np.size(t), 1))
    get_sun = lambda t: np.tile([149.6e9, 0, 0], (np.size(t), 1))

    def propagate(state):
        augmented = np.r_[state, np.eye(6).reshape(-1, order="F")]
        return propagate_augmented_state(
            times, augmented, config["mu_moon_m3_s2"], 0.0, 0.0,
            get_earth, get_sun, rtol=config["rtol"], atol=config["atol"],
        )[:, :6]

    truth = propagate(np.asarray(config["initial_state_m_mps"]))
    stations = tuple(Station(
        name=f"Synthetic {i}", lat_deg=lat, lon_deg=lon, alt_m=0.0,
        color_rgb=(0.0, 0.0, 0.0), sigma_range_m=config["sigma_range_m"],
        sigma_angle_rad=config["sigma_angle_rad"],
    ) for i, (lat, lon) in enumerate(config["stations_lat_lon_deg"], 1))
    geometry = PassGeometry(
        t_s=times, earth_pos_mci_m=get_earth(times),
        earth_vel_mci_mps=np.zeros((times.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None], times.size, axis=0),
        stations=stations, measurement_type="position",
    )
    observations = np.asarray([
        [t, 0, 0, 0, station, index, 1]
        for index, t in enumerate(times, 1)
        for station in range(1, len(stations) + 1)
    ], dtype=float)
    _, clean, _ = compute_position_residuals_analytic(truth, observations, geometry)
    sigma = np.array([config["sigma_range_m"], *[config["sigma_angle_rad"]] * 2])
    observations[:, 1:4] = clean
    if config["noise_enabled"]:
        observations[:, 1:4] += np.random.default_rng(config["seed"]).normal(size=clean.shape) * sigma
    return times, truth, geometry, observations, propagate, get_earth, get_sun


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    times, truth, geo, obs, propagate, earth, sun = build_case(config)
    initial = truth[0] + np.asarray(config["initial_error_m_mps"])
    initial_history = propagate(initial)
    histories, residuals, metrics = {}, {}, {}
    for label, estimator in (("BLS-LM", estimate_position_bls_lm),
                             ("SRIF", estimate_position_srif)):
        started = perf_counter()
        estimate, reason, stats = estimator(
            times, obs.copy(), initial.copy(), geo, config["mu_moon_m3_s2"],
            0.0, 0.0, earth, sun, max_iter=config["max_iter"],
            rtol=config["rtol"], atol=config["atol"], return_posterior=True,
        )
        elapsed = perf_counter() - started
        history = propagate(estimate[:6])
        residual = compute_position_residuals_analytic(history, obs, geo)[0].reshape(-1, 3)
        histories[label], residuals[label] = history, residual
        covariance = stats.posterior_covariance
        metrics[label] = {
            "stop_reason": reason, "iterations": stats.iterations,
            "initial_epoch_position_error_m": float(np.linalg.norm(estimate[:3] - truth[0, :3])),
            "arc_end_position_error_m": float(np.linalg.norm(history[-1, :3] - truth[-1, :3])),
            "arc_end_velocity_error_mps": float(np.linalg.norm(history[-1, 3:] - truth[-1, 3:])),
            "residual_rms_m_rad_rad": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
            "rank": int(stats.rank), "condition_number": float(stats.condition_number),
            "runtime_s": elapsed,
            "posterior_covariance_finite": bool(covariance is not None and np.isfinite(covariance).all()),
        }

    write_csv(output / "observations.csv", ["time_s", "range_m", "az_rad", "el_rad",
              "station_id", "epoch_index", "arc_id"], obs)
    rows = []
    for label, history in {"Initial guess": initial_history, **histories}.items():
        for t, state, reference in zip(times, history, truth):
            rows.append([label, t, *state, *reference,
                         np.linalg.norm(state[:3] - reference[:3])])
    write_csv(output / "states.csv", ["estimator", "time_s", "x_m", "y_m", "z_m",
        "vx_mps", "vy_mps", "vz_mps", "truth_x_m", "truth_y_m", "truth_z_m",
        "truth_vx_mps", "truth_vy_mps", "truth_vz_mps", "position_error_m"], rows)
    write_csv(output / "residuals.csv", ["estimator", "time_s", "station_id",
              "range_m", "az_rad", "el_rad"],
              ([label, row[0], row[4], *res] for label, values in residuals.items()
               for row, res in zip(obs, values)))

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
        for label, history in {"Initial guess": initial_history, **histories}.items():
            ax.semilogy(times / 60, np.linalg.norm(history[:, :3] - truth[:, :3], axis=1),
                        label=label, color=COLORS.get(label, "#777777"),
                        linestyle={"Initial guess": ":", "BLS-LM": "-", "SRIF": "--"}[label])
        ax.set(xlabel="Elapsed time (min)", ylabel="Position error against same-model truth (m)",
               title="One synthetic arc · identical measurements and initial guess")
        ax.legend()
        # Reveal the measured curves over time; this is a scientific plot
        # animation, not a screen recording or a simulated convergence history.
        lines = ax.lines[:]
        values = [(np.asarray(line.get_xdata()), np.asarray(line.get_ydata())) for line in lines]
        def reveal(end):
            for line, (x, y) in zip(lines, values):
                line.set_data(x[:end], y[:end])
            return lines
        animation = FuncAnimation(fig, reveal, frames=range(1, len(times) + 1, 2), blit=False)
        animation.save(output / "fit-preview.gif", writer=PillowWriter(fps=6), dpi=100)
        reveal(len(times))
        save_figure(fig, output, "position-error")
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.2), sharex=True, constrained_layout=True)
        for channel, (axis, unit) in enumerate(zip(axes, ("Range (m)", "Azimuth (arcsec)", "Elevation (arcsec)"))):
            scale = 1 if channel == 0 else 180 / np.pi * 3600
            for label, values in residuals.items():
                axis.scatter(obs[:, 0] / 60, values[:, channel] * scale, label=label,
                             color=COLORS[label], s=12, marker="o" if label == "BLS-LM" else "x", alpha=0.6)
            axis.axhline(0, color="#777777", linewidth=0.6)
            axis.set_ylabel(unit)
        axes[0].set_title(f"Post-fit residuals · noise={config['noise_enabled']}, seed {config['seed']}")
        axes[0].legend(ncol=2)
        axes[-1].set_xlabel("Elapsed time (min)")
        save_figure(fig, output, "residuals")

    metadata = {
        "schema_version": 1, "config": config, "config_sha256": sha256(config_path),
        "source": source_identity(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "processor": platform.processor(),
                        **{name: version(name) for name in ("numpy", "scipy", "matplotlib")}},
        "scope": {"truth_level": 0, "independent_arcs": 1, "independent_seeds": 1,
                  "force_model": "lunar point mass only", "station_frame": "fixed identity transform",
                  "visibility": "all scheduled observations retained; no elevation/occultation screening",
                  "light_time": False, "aberration": False, "explicit_prior": None,
                  "comparison": "educational paired fit; no performance ranking or operational claim"},
        "initial_position_error_m": float(np.linalg.norm(initial[:3] - truth[0, :3])),
        "observations": len(obs), "metrics": metrics,
    }
    metadata["artifact_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir())
                                    if p.is_file() and p.suffix in (".csv", ".png", ".svg", ".gif")}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, item in metrics.items():
        print(f"{label}: initial-epoch error {item['initial_epoch_position_error_m']:.3f} m; "
              f"arc-end error {item['arc_end_position_error_m']:.3f} m; {item['stop_reason']}")
    print(f"Wrote {output.resolve()} (Level 0 / one seed / educational example)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    args = parser.parse_args()
    run(args.config, args.output)
