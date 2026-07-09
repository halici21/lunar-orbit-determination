"""Phase 12B — real GRAIL model inventory, loader sanity, propagation sweep.

Validates the REAL GRAIL spherical-harmonic files relocated in Step 1 to the
gitignored ``data/gravity/<model>/`` tree, using the production loader and
propagator UNCHANGED.  No production code is modified.

Model registry (from Step 1 provenance, label-verified):
  grgm660prim  GRGM660PRIM  660/660   DE421 PA  -> primary propagation target
  grgm1200l    GRGM1200L    1199/1199 (DE430 PA via GRGM1200A background)
                                       -> loader/metadata sanity ONLY
  gl1800f      GL1800F      1800/1800 DE440 PA  -> secondary propagation target

Kernel profiles (Step 3): DE421 = the project default set
(``load_spice_kernels(clear=True)``); DE440 = a script-local MINIMAL list
{naif0012.tls.txt, moon_de440_220930.txt, moon_pa_de440_200625.bpc}.
``de440.bsp`` and ``pck00011.tpc.txt`` are deliberately excluded: pxform on a
binary-PCK frame needs orientation data only.  ``spice.kclear()`` runs before
EVERY profile load — moon_080317.tf and moon_de440_220930.txt both define the
bare ``MOON_PA`` alias (last furnsh wins), so the pools are never mixed and
every sampling call passes an EXPLICIT versioned frame name
(``MOON_PA_DE421`` / ``MOON_PA_DE440``) with ``load_kernels=False``.

Key loader fact this script works around: ``_parse_shadr`` has no early exit,
so parse cost is nmax-independent; each model file is loaded ONCE at the
highest target nmax and truncated in memory (script-local ``truncate_model``,
bit-identical vs direct load — verified in Step 2).

Stages:
  --inventory  directory/file/label presence, sizes, SHA256 recompute vs the
               Step 1 SHA256SUMS.txt records.
  --loader     one sanity load per model (nmax=8), timed; GRGM660PRIM only:
               one higher load (nmax=64) + bit-exact truncate_model check.
  --sweep      nmax truncation sweep, 1 orbit + 1 day (60 s rotation cadence):
               GRGM660PRIM 8/16/32/64 (+128 optional, runtime-gated) and
               GL1800F 8/16/32/64/128 (+256 optional, runtime-gated); plus the
               cross-model comparison at nmax=64.
  --guards     real-model guard checks: C20+j2_moon double-count ValueError,
               STM/augmented ValueError, Earth-J2 composability run.
  (default: inventory + loader)
Deferred: --plots (refuses to run).

Outputs (git-ignored, summary only — no dense trajectories):
  results/phase12b/phase12b_real_grail_metadata.json   (cumulative store)
  results/phase12b/phase12b_real_grail_inventory.md
  results/phase12b/phase12b_real_grail_truncation_sweep.csv
  results/phase12b/phase12b_real_grail_sweep_report.md
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from lunar_od.constants import (  # noqa: E402
    J2_EARTH_UNNORMALIZED, J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2,
    MU_SUN_M3S2, R_MOON_M,
)
from lunar_od.gravity_harmonics import SphericalHarmonicGravityModel  # noqa: E402
from lunar_od.gravity_model_loader import (  # noqa: E402
    describe_model, load_lunar_gravity_model, resolve_gravity_dir,
)

MU_M, MU_E, MU_S = MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2
J2000_JD = 2451545.0
OUT = ROOT / "results" / "phase12b"
JSON_PATH = OUT / "phase12b_real_grail_metadata.json"
INVENTORY_MD_PATH = OUT / "phase12b_real_grail_inventory.md"
SWEEP_CSV_PATH = OUT / "phase12b_real_grail_truncation_sweep.csv"
SWEEP_MD_PATH = OUT / "phase12b_real_grail_sweep_report.md"

SANITY_NMAX = 8
CADENCE_S = 60.0                    # Phase 13C accepted default
OPT_RUNTIME_GATE_S = 300.0          # per-window projected cap for optional nmax

# Step 3 planned single-load targets (slice down from these; parse cost is
# nmax-independent so today's measured load time predicts the Step 3 cost).
MODELS = {
    "grgm660prim": {
        "tab": "gggrx_0660pm_sha.tab",
        "lbl": "gggrx_0660pm_sha.lbl",
        "role": "primary propagation target (Step 3)",
        "frame_note": (
            "DE421 PA (label-declared) -> SPICE 'MOON_PA_DE421'; exact match "
            "with the project default kernel profile"
        ),
        "step3_load_nmax": 128,
        "step3_sweep": "8/16/32/64 (+128 optional, runtime-gated)",
        "high_load_nmax": 64,       # loaded in --loader for the truncation check
        "truncation_check": True,
    },
    "grgm1200l": {
        "tab": "gggrx_1200l_sha.tab",
        "lbl": "gggrx_1200l_sha.lbl",
        "role": "loader/metadata sanity ONLY (no propagation, no kernel)",
        "frame_note": (
            "no ephemeris declaration in label; inherits DE430 PA via the "
            "GRGM1200A background model — NOT a propagation target"
        ),
        "step3_load_nmax": None,
        "step3_sweep": "none",
        "high_load_nmax": None,
        "truncation_check": False,
    },
    "gl1800f": {
        "tab": "jggrx_1800f_sha.tab",
        "lbl": "jggrx_1800f_sha.lbl",
        "role": "secondary propagation target (Step 3)",
        "frame_note": (
            "DE440 PA (label-declared) -> SPICE 'MOON_PA_DE440'; script-local "
            "minimal kernel profile (kclear per profile switch)"
        ),
        "step3_load_nmax": 256,
        "step3_sweep": "8/16/32/64/128 (+256 optional, runtime-gated)",
        "high_load_nmax": None,     # 198 MB: parse exactly once per invocation
        "truncation_check": False,
    },
}

# Minimal DE440 rotation profile: pxform("J2000", "MOON_PA_DE440", et) needs
# the frame definitions + the binary-PCK orientation data (+ LSK, defensive).
# de440.bsp (translational) and pck00011.tpc (text PCK, superseded by the bpc
# for the Moon) are DELIBERATELY excluded to keep the pool surface minimal.
DE440_KERNELS = (
    "naif0012.tls.txt",
    "moon_de440_220930.txt",
    "moon_pa_de440_200625.bpc",
)

SWEEP_SPECS = {
    "grgm660prim": {
        "profile": "de421",
        "frame": "MOON_PA_DE421",
        "nmax_mandatory": (8, 16, 32, 64),
        "nmax_optional": 128,
        "load_nmax": 128,
    },
    "gl1800f": {
        "profile": "de440",
        "frame": "MOON_PA_DE440",
        "nmax_mandatory": (8, 16, 32, 64, 128),
        "nmax_optional": 256,
        "load_nmax": 256,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sums(path: Path) -> dict[str, str]:
    """Parse ``<hash>  <name>`` lines from a SHA256SUMS.txt file."""
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1]] = parts[0].lower()
    return sums


# ---------------------------------------------------------------------------
# Script-local truncation helper (production library deliberately untouched)
# ---------------------------------------------------------------------------
def truncate_model(
    model: SphericalHarmonicGravityModel, nmax: int, mmax: int | None = None
) -> SphericalHarmonicGravityModel:
    """Derive a lower-degree model by slicing the coefficient arrays.

    Rows n <= nmax only ever hold orders m <= n, so the square slice loses no
    retained coefficient.  ``dataclasses.replace`` re-runs ``__post_init__``,
    so the production shape/sign guards still apply to the result.
    """
    nmax = int(nmax)
    if nmax > model.nmax:
        raise ValueError(f"cannot truncate up: nmax={nmax} > model.nmax={model.nmax}.")
    mmax_eff = min(model.mmax, nmax) if mmax is None else int(mmax)
    metadata = dict(model.metadata)
    metadata["script_truncated_from_nmax"] = model.nmax
    return dataclasses.replace(
        model,
        cbar=model.cbar[: nmax + 1, : nmax + 1].copy(),
        sbar=model.sbar[: nmax + 1, : nmax + 1].copy(),
        nmax=nmax,
        mmax=mmax_eff,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Stage: inventory
# ---------------------------------------------------------------------------
def run_inventory(gravity_dir: Path) -> dict:
    print(f"[inventory] gravity dir: {gravity_dir}")
    inventory: dict = {"gravity_dir": str(gravity_dir), "models": {}}
    for key, spec in MODELS.items():
        model_dir = gravity_dir / key
        entry: dict = {"dir": str(model_dir), "dir_exists": model_dir.is_dir()}
        for label, name in (
            ("tab", spec["tab"]), ("lbl", spec["lbl"]),
            ("source", "SOURCE.txt"), ("sums", "SHA256SUMS.txt"),
        ):
            path = model_dir / name
            entry[f"{label}_file"] = name
            entry[f"{label}_present"] = path.is_file()
            entry[f"{label}_size_bytes"] = path.stat().st_size if path.is_file() else None
        if entry["sums_present"]:
            recorded = _read_sums(model_dir / "SHA256SUMS.txt")
            for label in ("tab", "lbl"):
                name = spec[label]
                if entry[f"{label}_present"]:
                    computed = _sha256_file(model_dir / name)
                    entry[f"{label}_sha256"] = computed
                    entry[f"{label}_sha256_match"] = recorded.get(name) == computed
        inventory["models"][key] = entry
        status = "OK" if all(
            entry.get(f"{lbl}_present") for lbl in ("tab", "lbl", "source", "sums")
        ) and entry.get("tab_sha256_match") and entry.get("lbl_sha256_match") else "PROBLEM"
        print(f"[inventory] {key:12s} {status}  tab={entry['tab_size_bytes']} B  "
              f"sha_match(tab/lbl)={entry.get('tab_sha256_match')}/{entry.get('lbl_sha256_match')}")
    return inventory


# ---------------------------------------------------------------------------
# Stage: loader sanity
# ---------------------------------------------------------------------------
def _model_report(key: str, spec: dict, model: SphericalHarmonicGravityModel,
                  load_seconds: float) -> dict:
    cbar, sbar = model.cbar, model.sbar
    cbar20 = float(cbar[2, 0])
    j2_derived = -math.sqrt(5.0) * cbar20
    tesseral = bool(np.any(cbar[:, 1:] != 0.0) or np.any(sbar[:, 1:] != 0.0))
    return {
        "model_key": key,
        "role": spec["role"],
        "frame_note": spec["frame_note"],
        "canonical_file": model.metadata["source_file"],
        "sha256": model.metadata["sha256"],
        "detected_format": model.metadata["format"],
        "native_normalization": model.metadata["native_normalization"],
        "declared_file_degree": model.metadata["file_degree"],
        "declared_file_order": model.metadata["file_order"],
        "records_loaded_within_truncation": model.metadata["records_loaded"],
        "loaded_nmax": model.nmax,
        "loaded_mmax": model.mmax,
        "load_seconds_full_file_parse": round(load_seconds, 3),
        "gm_m3_s2": model.mu_m3_s2,
        "r_ref_m": model.r_ref_m,
        "r_ref_note": (
            f"model R_ref={model.r_ref_m:.1f} m is the MODEL's own reference "
            f"radius; distinct from R_MOON_M={R_MOON_M:.1f} m (never mixed)"
        ),
        "cbar20": cbar20,
        "j2_derived_minus_sqrt5_cbar20": j2_derived,
        "j2_constant_reference": J2_MOON_UNNORMALIZED,
        "j2_rel_diff_vs_constant": abs(j2_derived - J2_MOON_UNNORMALIZED) / J2_MOON_UNNORMALIZED,
        "cbar22": float(cbar[2, 2]),
        "sbar22": float(sbar[2, 2]),
        "tesseral_m_gt_0_present": tesseral,
        "all_coefficients_finite": bool(np.isfinite(cbar).all() and np.isfinite(sbar).all()),
        "c00_in_array": float(cbar[0, 0]),
        "c00_note": (
            "SHADR GRAIL files carry no degree-0 record (implicit central term); "
            "0.0 here means 'absent from file', engine uses n>=2 only"
        ),
        "degree1_row": [float(cbar[1, 0]), float(cbar[1, 1]), float(sbar[1, 1])],
        "degree1_note": "validated ~0 by the loader at load time (raises otherwise)",
        "describe": describe_model(model),
    }


def run_loader(gravity_dir: Path) -> dict:
    loader: dict = {"sanity_nmax": SANITY_NMAX, "models": {}, "truncation_check": None}
    for key, spec in MODELS.items():
        tab_path = gravity_dir / key / spec["tab"]
        print(f"[loader] {key}: load nmax={SANITY_NMAX} (full-file parse) ...")
        t0 = time.perf_counter()
        model = load_lunar_gravity_model(tab_path, nmax=SANITY_NMAX)
        dt = time.perf_counter() - t0
        print(f"[loader] {key}: loaded in {dt:.2f} s -> {describe_model(model)}")
        loader["models"][key] = _model_report(key, spec, model, dt)

        if spec["truncation_check"]:
            high = spec["high_load_nmax"]
            print(f"[loader] {key}: load nmax={high} for the slice check ...")
            t0 = time.perf_counter()
            model_high = load_lunar_gravity_model(tab_path, nmax=high)
            dt_high = time.perf_counter() - t0
            sliced = truncate_model(model_high, SANITY_NMAX)
            check = {
                "high_nmax": high,
                "high_load_seconds": round(dt_high, 3),
                "cbar_bit_identical": bool(np.array_equal(sliced.cbar, model.cbar)),
                "sbar_bit_identical": bool(np.array_equal(sliced.sbar, model.sbar)),
                "nmax_equal": sliced.nmax == model.nmax,
                "mmax_equal": sliced.mmax == model.mmax,
                "gm_equal": sliced.mu_m3_s2 == model.mu_m3_s2,
                "r_ref_equal": sliced.r_ref_m == model.r_ref_m,
                "sha256_equal": sliced.metadata["sha256"] == model.metadata["sha256"],
                "file_degree_equal": sliced.metadata["file_degree"] == model.metadata["file_degree"],
            }
            if not (check["cbar_bit_identical"] and check["sbar_bit_identical"]):
                check["max_abs_cbar_diff"] = float(np.max(np.abs(sliced.cbar - model.cbar)))
                check["max_abs_sbar_diff"] = float(np.max(np.abs(sliced.sbar - model.sbar)))
            loader["truncation_check"] = check
            verdict = "PASS" if all(v is True for k, v in check.items()
                                    if k.endswith(("identical", "equal"))) else "FAIL"
            print(f"[loader] truncation slice check ({key}, {high}->{SANITY_NMAX}): {verdict}")
    return loader


# ---------------------------------------------------------------------------
# Step 3: kernel profiles + propagation harness (SPICE imported lazily so that
# --inventory/--loader stay runnable without SPICE/ephemeris)
# ---------------------------------------------------------------------------
def load_kernel_profile(profile: str) -> list[str]:
    """kclear + furnsh one profile.  DE421 and DE440 pools are NEVER mixed."""
    import spiceypy as spice
    from lunar_od.spice_loader import load_spice_kernels, resolve_kernel_dir

    spice.kclear()
    if profile == "de421":
        loaded = [p.name for p in load_spice_kernels(clear=True)]
    elif profile == "de440":
        kernel_dir = resolve_kernel_dir()
        loaded = []
        for name in DE440_KERNELS:
            path = kernel_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"DE440 profile kernel missing: {path}")
            spice.furnsh(str(path))
            loaded.append(name)
    else:
        raise ValueError(f"unknown kernel profile {profile!r}.")
    print(f"[kernel] profile '{profile}': kclear + furnsh {loaded}")
    return loaded


def rotation_pair(et0: float, t_end_s: float, cadence_s: float, frame: str):
    """Rotation grid [-margin, T+margin]; EXPLICIT versioned frame name only."""
    from lunar_od.lunar_frames import sample_moon_pa_rotations

    margin = max(2.0 * cadence_s, 120.0)
    t_grid = np.arange(-margin, t_end_s + margin + cadence_s / 2.0, cadence_s)
    rots = sample_moon_pa_rotations(et0, t_grid, frame=frame, load_kernels=False)
    return (t_grid, rots)


class _CountingGetter:
    """Counts RHS evaluations via the earth getter (called once per eval)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, t):
        self.calls += 1
        return self.fn(t)


def run_case(get_earth, get_sun, state0, teval, *, j2_moon=0.0, j2_earth=0.0,
             harmonic_model=None, harmonic_rotation=None) -> dict:
    """One propagation with runtime, RHS-eval count and altitude sanity."""
    from lunar_od.dynamics import propagate_state

    ge = _CountingGetter(get_earth)
    t0 = time.perf_counter()
    traj = propagate_state(
        teval, state0, MU_M, MU_E, MU_S, ge, get_sun,
        method="ADAMS", j2_moon=j2_moon, j2_earth=j2_earth,
        harmonic_model=harmonic_model, harmonic_rotation=harmonic_rotation,
    )
    runtime = time.perf_counter() - t0
    if not np.all(np.isfinite(traj)):
        raise RuntimeError("NaN/Inf in propagation output.")
    radii = np.linalg.norm(traj[:, :3], axis=1)
    min_radius = float(radii.min())
    return {
        "traj": traj,
        "runtime_s": float(runtime),
        "rhs_evals": int(ge.calls),
        "min_radius_m": min_radius,
        "min_altitude_m": min_radius - R_MOON_M,
        "surface_crossing": bool(min_radius <= R_MOON_M),
    }


def diff_metrics(traj, base) -> dict:
    dp = np.linalg.norm(traj[:, :3] - base[:, :3], axis=1)
    dv = np.linalg.norm(traj[:, 3:] - base[:, 3:], axis=1)
    return {
        "final_dpos_m": float(dp[-1]),
        "max_dpos_m": float(dp.max()),
        "rms_dpos_m": float(np.sqrt(np.mean(dp ** 2))),
        "final_dvel_mps": float(dv[-1]),
    }


def window_setup(window: str):
    """teval/ephemeris/initial state, continuous with the phase6/13C campaign."""
    from phase6_scenario_comparison import initial_state, load_ephemeris

    if window == "orbit":
        a = R_MOON_M + 100e3
        period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_M)      # ~7067 s
        t_end = math.ceil(period / 60.0) * 60.0
        out_step = 60.0
    elif window == "day1":
        t_end = 86400.0
        out_step = 600.0
    else:
        raise ValueError(f"unknown window {window!r}.")
    teval = np.arange(0.0, t_end + 1.0, out_step)
    eph, first_jd = load_ephemeris(t_end + 600.0)
    et0 = (first_jd - J2000_JD) * 86400.0
    return teval, t_end, eph, et0, initial_state()


# ---------------------------------------------------------------------------
# Stage: sweep
# ---------------------------------------------------------------------------
def run_sweep(gravity_dir: Path) -> dict:
    sweep: dict = {
        "cadence_s": CADENCE_S,
        "margin_policy": "max(2*cadence, 120 s)",
        "optional_runtime_gate_s": OPT_RUNTIME_GATE_S,
        "profile_kernels": {},
        "deliberate_exclusions": (
            "de440.bsp and pck00011.tpc.txt excluded from the DE440 rotation "
            "profile (pxform needs orientation data only); moon_de440_250416.tf "
            "not downloaded (moon_de440_220930.txt provides all frame "
            "definitions); GRGM1200L not propagated (no DE430 PA kernel exists; "
            "loader-sanity scope only)"
        ),
        "translational_ephemeris_note": (
            "Earth/Sun third-body positions come from the phase6 DE421-derived "
            ".mat ephemeris in ALL runs, including GL1800F+DE440 rotations; "
            "the DE421<->DE440 translational difference is negligible for "
            "third-body tides but is declared here for the record"
        ),
        "model_loads": {},
        "windows": {},
    }

    big: dict[str, SphericalHarmonicGravityModel] = {}
    for key, spec in SWEEP_SPECS.items():
        tab_path = gravity_dir / key / MODELS[key]["tab"]
        t0 = time.perf_counter()
        big[key] = load_lunar_gravity_model(tab_path, nmax=spec["load_nmax"])
        dt = time.perf_counter() - t0
        sweep["model_loads"][key] = {
            "load_nmax": spec["load_nmax"], "load_seconds": round(dt, 3),
            "describe": describe_model(big[key]),
        }
        print(f"[sweep] {key}: single load at nmax={spec['load_nmax']} in {dt:.2f} s "
              f"(lower nmax via truncate_model slices)")

    for window in ("orbit", "day1"):
        teval, t_end, eph, et0, s0 = window_setup(window)
        print(f"[sweep:{window}] t_end={t_end:.0f} s, {teval.size} epochs")
        wdata: dict = {"t_end_s": t_end, "n_epochs": int(teval.size),
                       "rows": [], "optional_decisions": {}, "rotation_grids": {}}
        traj_at_64: dict[str, np.ndarray] = {}

        for key, spec in SWEEP_SPECS.items():
            sweep["profile_kernels"][spec["profile"]] = load_kernel_profile(spec["profile"])
            pair = rotation_pair(et0, t_end, CADENCE_S, spec["frame"])
            wdata["rotation_grids"][key] = {
                "frame": spec["frame"], "cadence_s": CADENCE_S,
                "t_grid_range_s": [float(pair[0][0]), float(pair[0][-1])],
                "n_samples": int(pair[0].size),
            }

            prev_nmax = None
            prev_traj = None
            runtime_by_nmax: dict[int, float] = {}

            def run_one(nmax: int, optional: bool) -> None:
                nonlocal prev_nmax, prev_traj
                model_n = truncate_model(big[key], nmax)
                res = run_case(eph.earth_position, eph.sun_position, s0, teval,
                               harmonic_model=model_n, harmonic_rotation=pair)
                runtime_by_nmax[nmax] = res["runtime_s"]
                row = {
                    "model": key, "frame": spec["frame"], "window": window,
                    "nmax": nmax, "mmax": model_n.mmax, "optional": optional,
                    "cadence_s": CADENCE_S,
                    "runtime_s": round(res["runtime_s"], 3),
                    "rhs_evals": res["rhs_evals"],
                    "min_radius_m": round(res["min_radius_m"], 1),
                    "min_altitude_m": round(res["min_altitude_m"], 1),
                    "surface_crossing": res["surface_crossing"],
                    "nan": False,
                    "vs_nmax": prev_nmax,
                }
                if prev_traj is not None:
                    row.update(diff_metrics(res["traj"], prev_traj))
                wdata["rows"].append(row)
                if nmax == 64:
                    traj_at_64[key] = res["traj"]
                print(f"  {key} nmax={nmax:3d}{' (opt)' if optional else '      '} "
                      f"runtime {res['runtime_s']:7.2f} s  rhs {res['rhs_evals']:7d}  "
                      f"min_alt {res['min_altitude_m']/1e3:7.2f} km  "
                      f"{'SURFACE-CROSSING!' if res['surface_crossing'] else ''}"
                      + (f"  dpos_final {row['final_dpos_m']:.3e} m vs n{prev_nmax}"
                         if prev_traj is not None else ""))
                prev_nmax, prev_traj = nmax, res["traj"]

            for nmax in spec["nmax_mandatory"]:
                run_one(nmax, optional=False)

            opt = spec["nmax_optional"]
            highest = spec["nmax_mandatory"][-1]
            projected = runtime_by_nmax[highest] * (opt / highest) ** 2
            if projected <= OPT_RUNTIME_GATE_S:
                wdata["optional_decisions"][key] = {
                    "nmax": opt, "ran": True,
                    "projected_runtime_s": round(projected, 1),
                    "basis": f"runtime(nmax={highest}) * ({opt}/{highest})^2",
                }
                run_one(opt, optional=True)
            else:
                wdata["optional_decisions"][key] = {
                    "nmax": opt, "ran": False,
                    "projected_runtime_s": round(projected, 1),
                    "basis": f"runtime(nmax={highest}) * ({opt}/{highest})^2",
                    "reason": f"projected {projected:.0f} s exceeds the "
                              f"{OPT_RUNTIME_GATE_S:.0f} s per-window gate",
                }
                print(f"  {key} nmax={opt} SKIPPED (projected {projected:.0f} s "
                      f"> gate {OPT_RUNTIME_GATE_S:.0f} s)")

        if len(traj_at_64) == 2:
            cross = diff_metrics(traj_at_64["grgm660prim"], traj_at_64["gl1800f"])
            wdata["cross_model_nmax64"] = {
                **cross,
                "note": (
                    "GRGM660PRIM@64 (MOON_PA_DE421) vs GL1800F@64 (MOON_PA_DE440): "
                    "the difference mixes (1) the coefficient-model difference, "
                    "(2) the DE421 vs DE440 orientation difference, and (3) both "
                    "runs sharing the phase6 DE421-derived .mat translational "
                    "ephemeris — the three effects are NOT separated here"
                ),
            }
        sweep["windows"][window] = wdata

    surface = [r for w in sweep["windows"].values() for r in w["rows"] if r["surface_crossing"]]
    if surface:
        print(f"[sweep] WARNING: surface crossing in {len(surface)} run(s) — see report")
    return sweep


# ---------------------------------------------------------------------------
# Stage: guards (real GRGM660PRIM model, DE421 profile)
# ---------------------------------------------------------------------------
def run_guards(gravity_dir: Path) -> dict:
    from lunar_od.dynamics import propagate_augmented_state

    tab_path = gravity_dir / "grgm660prim" / MODELS["grgm660prim"]["tab"]
    model8 = load_lunar_gravity_model(tab_path, nmax=SANITY_NMAX)
    guards: dict = {"model": describe_model(model8), "frame": "MOON_PA_DE421"}

    teval, t_end, eph, et0, s0 = window_setup("orbit")
    guards["kernels"] = load_kernel_profile("de421")
    pair = rotation_pair(et0, t_end, CADENCE_S, "MOON_PA_DE421")

    # 1) double-count: real model has Cbar20 != 0, so j2_moon must be 0
    try:
        run_case(eph.earth_position, eph.sun_position, s0, teval,
                 j2_moon=J2_MOON_UNNORMALIZED,
                 harmonic_model=model8, harmonic_rotation=pair)
        guards["double_count"] = {"raised": False, "verdict": "FAIL"}
    except ValueError as exc:
        message = str(exc)
        guards["double_count"] = {
            "raised": True, "message": message,
            "message_ok": "count J2 twice" in message,
            "verdict": "PASS" if "count J2 twice" in message else "FAIL",
        }
    print(f"[guards] double-count: {guards['double_count']['verdict']}")

    # 2) STM/augmented refusal with a real model
    x_aug0 = np.concatenate([s0, np.eye(6).flatten(order="F")])
    try:
        propagate_augmented_state(teval, x_aug0, MU_M, MU_E, MU_S,
                                  eph.earth_position, eph.sun_position,
                                  harmonic_model=model8)
        guards["stm_refusal"] = {"raised": False, "verdict": "FAIL"}
    except ValueError as exc:
        message = str(exc)
        expected = "lunar harmonics gradient not implemented"
        guards["stm_refusal"] = {
            "raised": True, "message": message,
            "message_ok": expected in message,
            "verdict": "PASS" if expected in message else "FAIL",
        }
    print(f"[guards] STM refusal: {guards['stm_refusal']['verdict']}")

    # 3) Earth J2 stays composable with real lunar harmonics
    res_no = run_case(eph.earth_position, eph.sun_position, s0, teval,
                      harmonic_model=model8, harmonic_rotation=pair)
    res_ej2 = run_case(eph.earth_position, eph.sun_position, s0, teval,
                       j2_earth=J2_EARTH_UNNORMALIZED,
                       harmonic_model=model8, harmonic_rotation=pair)
    effect = diff_metrics(res_ej2["traj"], res_no["traj"])
    guards["earth_j2_composable"] = {
        "window": "orbit", "nmax": SANITY_NMAX,
        "runtime_s": round(res_ej2["runtime_s"], 3),
        "rhs_evals": res_ej2["rhs_evals"],
        "min_altitude_m": round(res_ej2["min_altitude_m"], 1),
        "finite": True,
        **effect,
        "nonzero_effect": effect["final_dpos_m"] > 0.0,
        "verdict": "PASS" if effect["final_dpos_m"] > 0.0 else "FAIL",
    }
    print(f"[guards] Earth-J2 composability: {guards['earth_j2_composable']['verdict']} "
          f"(final dpos {effect['final_dpos_m']:.3e} m over 1 orbit)")
    return guards


# ---------------------------------------------------------------------------
# Outputs (cumulative JSON store; MD/CSV regenerated from the merged store)
# ---------------------------------------------------------------------------
def _load_store() -> dict:
    if JSON_PATH.exists():
        try:
            return json.loads(JSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_inventory_md(store: dict) -> None:
    inventory = store.get("inventory")
    loader = store.get("loader")
    if inventory is None and loader is None:
        return
    lines = [
        "# Phase 12B — Real GRAIL Inventory & Loader Sanity",
        "",
        f"- store updated (UTC): {store.get('generated_utc')}",
        "- scope: inventory + loader sanity (sweep/guards in the sweep report)",
        "",
    ]
    if inventory is not None:
        lines += [
            "## Inventory",
            "",
            f"- gravity dir (resolver): `{inventory['gravity_dir']}`",
            "",
            "| model | tab (bytes) | lbl | SOURCE.txt | SHA256SUMS.txt | sha match tab/lbl |",
            "|---|---|---|---|---|---|",
        ]
        for key, e in inventory["models"].items():
            lines.append(
                f"| {key} | {e['tab_present']} ({e['tab_size_bytes']}) | {e['lbl_present']} "
                f"| {e['source_present']} | {e['sums_present']} "
                f"| {e.get('tab_sha256_match')} / {e.get('lbl_sha256_match')} |"
            )
        lines.append("")
    if loader is not None:
        lines += ["## Loader sanity (production loader, unchanged)", ""]
        for key, r in loader["models"].items():
            lines += [
                f"### {key}",
                "",
                f"- `{r['describe']}`",
                f"- role: {r['role']}",
                f"- frame: {r['frame_note']}",
                f"- declared degree/order: {r['declared_file_degree']}/{r['declared_file_order']}"
                f"; loaded nmax/mmax: {r['loaded_nmax']}/{r['loaded_mmax']}"
                f"; records within truncation: {r['records_loaded_within_truncation']}",
                f"- GM = {r['gm_m3_s2']:.9e} m^3/s^2; R_ref = {r['r_ref_m']:.1f} m "
                f"(R_MOON_M = {R_MOON_M:.1f} m — distinct, never mixed)",
                f"- Cbar20 = {r['cbar20']:.9e}; derived J2 = {r['j2_derived_minus_sqrt5_cbar20']:.9e} "
                f"(constants J2_MOON = {r['j2_constant_reference']:.9e}, "
                f"rel diff {r['j2_rel_diff_vs_constant']:.3e})",
                f"- Cbar22 = {r['cbar22']:.9e}, Sbar22 = {r['sbar22']:.9e}; "
                f"tesseral m>0 present: {r['tesseral_m_gt_0_present']}",
                f"- finite: {r['all_coefficients_finite']}; degree-1 row {r['degree1_row']} "
                f"({r['degree1_note']})",
                f"- full-file parse time at nmax={loader['sanity_nmax']}: "
                f"{r['load_seconds_full_file_parse']} s (parse cost is nmax-independent)",
                "",
            ]
        check = loader.get("truncation_check")
        if check is not None:
            lines += [
                "## Truncation slice check (grgm660prim)",
                "",
                f"- direct load nmax={loader['sanity_nmax']} vs truncate_model"
                f"({check['high_nmax']} -> {loader['sanity_nmax']}):",
                f"- cbar bit-identical: {check['cbar_bit_identical']}; "
                f"sbar bit-identical: {check['sbar_bit_identical']}; "
                f"nmax/mmax/GM/R_ref/sha256/file_degree equal: "
                f"{check['nmax_equal']}/{check['mmax_equal']}/{check['gm_equal']}"
                f"/{check['r_ref_equal']}/{check['sha256_equal']}/{check['file_degree_equal']}",
                f"- high load (nmax={check['high_nmax']}) parse time: "
                f"{check['high_load_seconds']} s",
                "",
            ]
    INVENTORY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] wrote {INVENTORY_MD_PATH}")


_CSV_FIELDS = [
    "model", "frame", "window", "nmax", "mmax", "optional", "cadence_s",
    "runtime_s", "rhs_evals", "min_radius_m", "min_altitude_m",
    "surface_crossing", "nan", "vs_nmax",
    "final_dpos_m", "max_dpos_m", "rms_dpos_m", "final_dvel_mps",
]


def _write_sweep_csv(sweep: dict) -> None:
    with SWEEP_CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for window in sweep["windows"].values():
            for row in window["rows"]:
                writer.writerow(row)
    print(f"[out] wrote {SWEEP_CSV_PATH}")


def _write_sweep_md(store: dict) -> None:
    sweep = store.get("sweep")
    guards = store.get("guards")
    if sweep is None and guards is None:
        return
    lines = [
        "# Phase 12B — Real GRAIL Propagation Sweep & Guard Validation",
        "",
        f"- store updated (UTC): {store.get('generated_utc')}",
        "",
    ]
    if sweep is not None:
        lines += [
            "## Kernel profiles (pool separation)",
            "",
            f"- DE421 profile: {sweep['profile_kernels'].get('de421')}",
            f"- DE440 profile (minimal): {sweep['profile_kernels'].get('de440')}",
            "- `spice.kclear()` before EVERY profile load; bare `MOON_PA` never "
            "used (moon_080317.tf and moon_de440_220930.txt both alias it — "
            "last furnsh would win); all sampling calls pass explicit "
            "`MOON_PA_DE421` / `MOON_PA_DE440` with `load_kernels=False`.",
            f"- deliberate exclusions: {sweep['deliberate_exclusions']}",
            f"- translational ephemeris: {sweep['translational_ephemeris_note']}",
            "",
            "## Model loads (load-once + slice)",
            "",
        ]
        for key, ml in sweep["model_loads"].items():
            lines.append(f"- {key}: single parse at nmax={ml['load_nmax']} in "
                         f"{ml['load_seconds']} s; `{ml['describe']}`")
        lines += ["", f"- rotation cadence: {sweep['cadence_s']} s; margin policy: "
                  f"{sweep['margin_policy']}; optional-nmax gate: projected runtime <= "
                  f"{sweep['optional_runtime_gate_s']} s per window", ""]
        for wname, wdata in sweep["windows"].items():
            lines += [
                f"## Window: {wname} (t_end = {wdata['t_end_s']:.0f} s, "
                f"{wdata['n_epochs']} epochs)",
                "",
                "| model | nmax | mmax | opt | runtime s | RHS evals | min alt km "
                "| cross | vs nmax | final dpos m | max dpos m | rms dpos m | final dvel m/s |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for r in wdata["rows"]:
                fdp = f"{r['final_dpos_m']:.3e}" if "final_dpos_m" in r else "-"
                mdp = f"{r['max_dpos_m']:.3e}" if "max_dpos_m" in r else "-"
                rdp = f"{r['rms_dpos_m']:.3e}" if "rms_dpos_m" in r else "-"
                fdv = f"{r['final_dvel_mps']:.3e}" if "final_dvel_mps" in r else "-"
                lines.append(
                    f"| {r['model']} | {r['nmax']} | {r['mmax']} "
                    f"| {'y' if r['optional'] else ''} | {r['runtime_s']} "
                    f"| {r['rhs_evals']} | {r['min_altitude_m']/1e3:.2f} "
                    f"| {'YES' if r['surface_crossing'] else 'no'} "
                    f"| {r['vs_nmax'] if r['vs_nmax'] is not None else '-'} "
                    f"| {fdp} | {mdp} | {rdp} | {fdv} |"
                )
            lines.append("")
            for key, dec in wdata["optional_decisions"].items():
                lines.append(
                    f"- optional nmax={dec['nmax']} for {key}: "
                    f"{'RAN' if dec['ran'] else 'SKIPPED'} "
                    f"(projected {dec['projected_runtime_s']} s via {dec['basis']}"
                    + (f"; {dec['reason']}" if not dec["ran"] else "") + ")"
                )
            for key, rg in wdata["rotation_grids"].items():
                lines.append(
                    f"- rotation grid {key}: frame={rg['frame']}, cadence={rg['cadence_s']} s, "
                    f"range=[{rg['t_grid_range_s'][0]:.0f}, {rg['t_grid_range_s'][1]:.0f}] s, "
                    f"{rg['n_samples']} samples"
                )
            cross = wdata.get("cross_model_nmax64")
            if cross:
                lines += [
                    "",
                    f"- cross-model nmax=64: final dpos {cross['final_dpos_m']:.3e} m, "
                    f"max {cross['max_dpos_m']:.3e} m, rms {cross['rms_dpos_m']:.3e} m, "
                    f"final dvel {cross['final_dvel_mps']:.3e} m/s",
                    f"  - NOTE: {cross['note']}",
                ]
            lines.append("")
    if guards is not None:
        dc, stm, ej2 = (guards["double_count"], guards["stm_refusal"],
                        guards["earth_j2_composable"])
        lines += [
            "## Guard validation (real GRGM660PRIM nmax=8, MOON_PA_DE421)",
            "",
            f"- model: `{guards['model']}`",
            f"- double-count (harmonics C20 + j2_moon!=0): {dc['verdict']} — "
            f"raised={dc['raised']}, message contains 'count J2 twice': "
            f"{dc.get('message_ok')}",
            f"- STM/augmented refusal: {stm['verdict']} — raised={stm['raised']}, "
            f"message contains 'lunar harmonics gradient not implemented': "
            f"{stm.get('message_ok')}",
            f"- Earth-J2 composability (1 orbit, nmax={ej2['nmax']}): {ej2['verdict']} — "
            f"finite={ej2['finite']}, final dpos vs Earth-J2-off = "
            f"{ej2['final_dpos_m']:.3e} m, runtime {ej2['runtime_s']} s, "
            f"RHS evals {ej2['rhs_evals']}, min alt {ej2['min_altitude_m']/1e3:.2f} km",
            "",
        ]
    SWEEP_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] wrote {SWEEP_MD_PATH}")


def write_outputs(store: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store["step"] = "12B (cumulative: inventory/loader/sweep/guards)"
    JSON_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"[out] wrote {JSON_PATH}")
    _write_inventory_md(store)
    if store.get("sweep") is not None:
        _write_sweep_csv(store["sweep"])
    _write_sweep_md(store)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", action="store_true", help="file/provenance inventory")
    parser.add_argument("--loader", action="store_true", help="loader sanity (nmax=8 per model)")
    parser.add_argument("--sweep", action="store_true", help="propagation truncation sweep")
    parser.add_argument("--guards", action="store_true", help="real-model guard validation")
    parser.add_argument("--plots", action="store_true", help="(deferred — refuses to run)")
    args = parser.parse_args(argv)

    if args.plots:
        print("[phase12b] --plots is deliberately not implemented in Step 3.")
        return 2

    explicit = args.inventory or args.loader or args.sweep or args.guards
    do_inventory = args.inventory or not explicit
    do_loader = args.loader or not explicit

    gravity_dir = resolve_gravity_dir()
    store = _load_store()
    if do_inventory:
        store["inventory"] = run_inventory(gravity_dir)
    if do_loader:
        store["loader"] = run_loader(gravity_dir)
    if args.sweep:
        store["sweep"] = run_sweep(gravity_dir)
    if args.guards:
        store["guards"] = run_guards(gravity_dir)
    write_outputs(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
