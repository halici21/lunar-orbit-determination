"""Visualize the optional light-time (CN) and stellar aberration (+S) corrections.

Regenerates ``docs/figures/aberration_corrections.png`` from the committed SPICE
fixture. Run from the project root (kernels must be available)::

    python python_port/examples/plot_aberration_corrections.py

Three panels:
  1. One-way light-time range correction  (|range_CN - range_geometric|)
  2. Stellar aberration angular shift      (local_mci vs spice_ssb, arcsec)
  3. Range invariance under +S             (|range_+S - range_CN|, pure rotation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import spiceypy as spice  # noqa: E402

from lunar_od import (  # noqa: E402
    MoonCenteredEphemeris,
    generate_position_measurements,
    load_spice_kernels,
    range_rate_stations,
)

C_LIGHT = 299792458.0
PYTHON_PORT = Path(__file__).resolve().parents[1]
FIXTURE = PYTHON_PORT / "fixtures" / "spice_snapshots.json"
OUT = PYTHON_PORT / "docs" / "figures" / "aberration_corrections.png"

NAVY = "#012855"
GOLD = "#c89a2b"
TEAL = "#0f9b8e"


def _load_fixture():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    truth = fx["truth_propagation"]
    meas = fx["position_measurements"]
    eph = MoonCenteredEphemeris(
        t_ephem_s=truth["t_ephem_s"],
        earth_pos_m=truth["earth_pos_grid_m"],
        sun_pos_m=truth["sun_pos_grid_m"],
        earth_vel_mps=truth["earth_vel_grid_mps"],
    )
    by = {s.name: s for s in range_rate_stations()}
    stations = [by[n] for n in meas["station_names"]]
    return fx, truth, meas, eph, stations


def _generate(fx, truth, meas, eph, stations, **kw):
    load_spice_kernels()
    try:
        _, _pg, clean = generate_position_measurements(
            np.asarray(meas["t_pass_s"], float),
            np.asarray(truth["state_history_mci_m_mps"], float),
            stations,
            meas["vis_mask_raw"],
            eph.earth_position,
            eph.earth_velocity,
            fx["et"],
            noise=False,
            **kw,
        )
        return clean
    finally:
        spice.kclear()


def _angular_sep_arcsec(a, b):
    """Great-circle separation between two (az, el) pointing sets, in arcsec."""
    az1, el1 = a[:, 2], a[:, 3]
    az2, el2 = b[:, 2], b[:, 3]
    cos_sep = np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    return np.degrees(sep) * 3600.0


def _earth_ssb_speed(et):
    load_spice_kernels()
    try:
        st, _ = spice.spkezr("EARTH", float(et), "J2000", "NONE", "SSB")
        return float(np.linalg.norm(np.asarray(st[3:6]) * 1000.0))
    finally:
        spice.kclear()


def main() -> int:
    if not FIXTURE.is_file():
        raise SystemExit(f"Fixture not found: {FIXTURE}")

    fx, truth, meas, eph, stations = _load_fixture()
    geom = _generate(fx, truth, meas, eph, stations, apply_light_time=False)
    cn = _generate(fx, truth, meas, eph, stations, apply_light_time=True)
    loc = _generate(
        fx, truth, meas, eph, stations,
        apply_light_time=True, apply_stellar_aberration=True,
        stellar_aberration_model="local_mci",
    )
    ssb = _generate(
        fx, truth, meas, eph, stations,
        apply_light_time=True, apply_stellar_aberration=True,
        stellar_aberration_model="spice_ssb",
    )

    n = cn.shape[0]
    idx = np.arange(1, n + 1)
    lt_range = np.abs(cn[:, 1] - geom[:, 1])                      # m
    loc_shift = _angular_sep_arcsec(loc, cn)                       # arcsec
    ssb_shift = _angular_sep_arcsec(ssb, cn)                       # arcsec
    loc_range = np.abs(loc[:, 1] - cn[:, 1]) + 1e-12               # m (floored for log)
    ssb_range = np.abs(ssb[:, 1] - cn[:, 1]) + 1e-12              # m

    v_ssb = _earth_ssb_speed(fx["et"])
    ssb_ceiling_as = np.degrees(v_ssb / C_LIGHT) * 3600.0         # |v_ssb|/c in arcsec

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888"})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    fig.suptitle(
        "Optional aberration corrections on range / azimuth / elevation "
        f"(SPICE fixture, n = {n} obs)",
        fontsize=13, fontweight="bold", color=NAVY,
    )

    # Panel 1: light-time range correction
    ax = axes[0]
    ax.bar(idx, lt_range, color=NAVY, width=0.7)
    ax.set_title("1.  One-way light time (CN)\nrange correction vs geometric",
                 fontsize=11, color=NAVY)
    ax.set_xlabel("observation #")
    ax.set_ylabel(r"$|\,\rho_{CN} - \rho_{geom}\,|$   [m]")
    ax.axhline(np.median(lt_range), color=GOLD, ls="--", lw=1.3,
               label=f"median {np.median(lt_range):.0f} m")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Panel 2: stellar aberration angular shift
    ax = axes[1]
    w = 0.4
    ax.bar(idx - w / 2, loc_shift, width=w, color=TEAL, label="local_mci")
    ax.bar(idx + w / 2, ssb_shift, width=w, color=GOLD, label="spice_ssb")
    ax.axhline(ssb_ceiling_as, color="#b00", ls=":", lw=1.4,
               label=f"|v_SSB|/c = {ssb_ceiling_as:.1f} arcsec")
    ax.set_title("2.  Stellar aberration (+S)\nazimuth/elevation shift",
                 fontsize=11, color=NAVY)
    ax.set_xlabel("observation #")
    ax.set_ylabel("angular shift  [arcsec]")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ratio = np.max(ssb_shift) / max(np.max(loc_shift), 1e-30)
    ax.text(0.03, 0.95, f"SSB / MCI  ≈ {ratio:.0f}×",
            transform=ax.transAxes, va="top", fontsize=9.5, color=NAVY,
            bbox=dict(boxstyle="round", fc="#eef2f8", ec="#ccc"))

    # Panel 3: range invariance under +S (pure rotation)
    ax = axes[2]
    ax.semilogy(idx, loc_range, "o-", color=TEAL, ms=4, label="local_mci")
    ax.semilogy(idx, ssb_range, "s-", color=GOLD, ms=4, label="spice_ssb")
    ax.axhline(1e-3, color="#b00", ls=":", lw=1.3, label="1 mm")
    ax.set_title("3.  Range invariance under +S\n(pure rotation preserves norm)",
                 fontsize=11, color=NAVY)
    ax.set_xlabel("observation #")
    ax.set_ylabel(r"$|\,\rho_{+S} - \rho_{CN}\,|$   [m]")
    ax.set_ylim(1e-10, 1e-1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(f"  light-time range corr: median {np.median(lt_range):.1f} m, "
          f"max {np.max(lt_range):.1f} m")
    print(f"  +S shift local_mci max: {np.max(loc_shift):.3f}\"  "
          f"spice_ssb max: {np.max(ssb_shift):.3f}\"  (ceiling {ssb_ceiling_as:.1f}\")")
    print(f"  +S range change max: {np.max(ssb_range):.2e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
