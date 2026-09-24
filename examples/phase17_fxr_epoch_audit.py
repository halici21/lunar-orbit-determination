"""PHASE 17-FXR s13-s18 - Earth-epoch runtime audit, offset proof, clamp
characterization, historical reproduction and four-way epoch/clamp attribution.

ANALYSIS SPACE ONLY.  No production file is modified.  Every design matrix is
produced by `phase17_r1m_core.build_range_arc`, i.e. by the production
propagator, two-way range model, Jacobian and K column; only the Earth
ephemeris handed to the MEASUREMENT model differs between the four arms:

  A  LEGACY      campaign table: epoch 504000 s early AND clamped (published)
  B  OFFSET      SPICE at the campaign ET0 (early), NOT clamped
  C  CLAMP       SPICE at the measurement ET, clamped at the table's last node
  D  CANONICAL   SPICE at the measurement ET over the whole arc (FXR fixture)

Configuration = the published R1O/R1O-D/R1O-R/GEO-G0 range baseline:
15 orbits, Goldstone + Madrid + Canberra, 90 s cadence.

Run:  PYTHONPATH="<worktree>;<od_covariance_campaign>/06_scripts"
      python examples/phase17_fxr_epoch_audit.py
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"
GEO_CACHE = ROOT / "results" / "phase17_geo_cache"
PHASE7_MANIFEST = Path(r"C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign/"
                       r"08_manifests/phase7_measurement_geometry_manifest.json")
EPHEMERIS_MAT = Path(r"C:/Users/erayh/Documents/Python/Grad/ephemeris_data.mat")

DSN3 = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
W15_ORBITS = 15.0
CADENCE_S = 90.0

ARMS = {
    "A_LEGACY_OFFSET_CLAMPED": ("EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED", "FIXTURE_PUBLISHED"),
    "B_OFFSET_EPOCH_UNCLAMPED": ("EARTH_FIXTURE_DIAG_OFFSET_UNCLAMPED", "FIXTURE_OFFSET_ONLY"),
    "C_CORRECT_EPOCH_CLAMPED": ("EARTH_FIXTURE_DIAG_CORRECT_EPOCH_CLAMPED", "FIXTURE_CLAMP_ONLY"),
    "D_CANONICAL": ("EARTH_FIXTURE_CANONICAL", "FIXTURE_CONSISTENT"),
}


def hdr(t: str) -> None:
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")


def build_arm(arm: str) -> dict:
    """One W15 arc under one Earth fixture; design matrices + metrics."""
    import phase17_geo_core as G
    import phase17_r1m_core as M

    fixture = getattr(M, ARMS[arm][0])
    _, t_orbit = M.campaign_epoch()
    arc = M.build_range_arc(0.0, W15_ORBITS * t_orbit, label=arm, station_filter=DSN3,
                            cadence_s=CADENCE_S, earth_fixture=fixture)
    m = G.information_metrics(arc.h_x0, arc.h_k, arc.w)
    ref = np.load(GEO_CACHE / ("%s.npz" % ARMS[arm][1]))
    same_shape = ref["h_x0"].shape == arc.h_x0.shape
    bitwise = bool(same_shape and all(np.array_equal(getattr(arc, k), ref[k])
                                      for k in ("h_x0", "h_k", "w", "obs")))
    pg = arc.pass_geo
    return dict(
        arm=arm, fixture=fixture, earth=arc.earth.describe(),
        n_obs=int(arc.n_obs), obs_t=arc.obs[:, 0].tolist(),
        obs_station=arc.obs[:, 2].astype(int).tolist(),
        pass_et0=float(pg.et0_s), pass_t=np.asarray(pg.t_s).tolist(),
        pass_earth_pos=np.asarray(pg.earth_pos_mci_m).tolist(),
        pass_earth_vel=np.asarray(pg.earth_vel_mci_mps).tolist(),
        nom48_sha=__import__("hashlib").sha256(np.ascontiguousarray(arc.nom48)).hexdigest(),
        bitwise_vs_geo_cache=bitwise, geo_cache=ARMS[arm][1],
        metrics={k: (v if not isinstance(v, np.generic) else float(v)) for k, v in m.items()},
    )


def epoch_reconstruction() -> dict:
    """s15: recover both epochs from the data files the code actually reads."""
    import h5py

    import phase17_r1m_core as M

    man = json.loads(PHASE7_MANIFEST.read_text())
    with h5py.File(EPHEMERIS_MAT, "r") as f:
        first_jd = float(np.array(f["first_jd"]).ravel()[0])
    et_table = (first_jd - 2451545.0) * 86400.0        # campaign module's ET0 formula
    et_scen, t_orbit = M.campaign_epoch()
    C = M._legacy_campaign_module()  # noqa: N806
    x0 = M.campaign_initial_state()
    return dict(
        scenario_et=et_scen,
        scenario_et_source="phase7_measurement_geometry_manifest.json:phase7_et0 "
                           "(phase17_r1m_core.campaign_epoch)",
        historical_earth_table_start_et=et_table,
        earth_table_source="ephemeris_data.mat:first_jd -> od_gravity_covariance_campaign.ET0",
        campaign_module_et0=float(C.ET0),
        campaign_module_et0_matches_mat=bool(float(C.ET0) == et_table),
        manifest_canonical_et0=float(man["canonical_et0"]),
        manifest_offset_s=float(man["offset_s"]),
        delta_et_seconds=et_scen - et_table,
        delta_et_days=(et_scen - et_table) / 86400.0,
        delta_matches_manifest_offset=bool(et_scen - et_table == float(man["offset_s"])),
        orbit_period_s=t_orbit,
        spacecraft_x0_equals_manifest_x0_truth_phase7=bool(
            np.array_equal(x0, np.asarray(man.get("x0_truth_phase7", x0 * np.nan), float)))
        if "x0_truth_phase7" in man else "FIELD_ABSENT",
        x0_truth_provenance=man["x0_truth_provenance"],
        scenario_utc=man["phase7_utc"],
    )


def direction_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def runtime_audit(arms: dict, recon: dict) -> dict:
    """s13/s15: what ET each geometric quantity is actually evaluated at."""
    import spiceypy as spice

    import phase17_r1m_core as M

    et_scen = recon["scenario_et"]
    et_tab = recon["historical_earth_table_start_et"]
    out = {}
    for key in ("A_LEGACY_OFFSET_CLAMPED", "D_CANONICAL"):
        a = arms[key]
        t = np.asarray(a["pass_t"])
        pos = np.asarray(a["pass_earth_pos"])
        vel = np.asarray(a["pass_earth_vel"])
        at_meas = np.array([M._spice_earth_state_mci(a["pass_et0"] + x) for x in t])
        at_table = np.array([M._spice_earth_state_mci(et_tab + x) for x in t])
        pre = t <= a["earth"]["t_max_s"] if key.startswith("A") else np.ones_like(t, bool)
        out[key] = dict(
            rotation_et_origin=a["pass_et0"],
            earth_et_origin=a["earth"]["et_origin"],
            earth_minus_rotation_origin_s=a["earth"]["et_origin"] - a["pass_et0"],
            max_abs_earth_pos_diff_vs_spice_at_measurement_et_m=float(
                np.max(np.abs(pos - at_meas[:, :3]))),
            max_abs_earth_pos_diff_vs_spice_at_table_et_m_before_clamp=float(
                np.max(np.abs(pos[pre] - at_table[pre, :3]))),
            max_abs_earth_vel_diff_vs_spice_at_measurement_et_mps=float(
                np.max(np.abs(vel - at_meas[:, 3:]))),
        )
    # LOS rotation the offset imposes, and Sun-Moon-Earth geometry
    t = np.asarray(arms["D_CANONICAL"]["pass_t"])
    pa = np.asarray(arms["A_LEGACY_OFFSET_CLAMPED"]["pass_earth_pos"])
    pd = np.asarray(arms["D_CANONICAL"]["pass_earth_pos"])
    ang = np.array([direction_deg(pa[i], pd[i]) for i in range(t.size)])

    def sun(et):
        return np.asarray(spice.spkezr("SUN", et, "J2000", "NONE", "MOON")[0][:3]) * 1e3

    phase_correct = direction_deg(sun(et_scen), M._spice_earth_state_mci(et_scen)[:3])
    phase_legacy = direction_deg(sun(et_scen), M._spice_earth_state_mci(et_tab)[:3])
    out["moon_to_earth_direction_error_deg"] = dict(
        at_t0=float(ang[0]), at_clamp=float(ang[np.searchsorted(t, 28260.0)]),
        at_arc_end=float(ang[-1]), max=float(ang.max()))
    out["sun_moon_earth_angle_deg"] = dict(
        correct_epoch=phase_correct, legacy_table_epoch=phase_legacy,
        note="angle at the Moon between the Moon->Sun and Moon->Earth directions; "
             "180 deg = Earth exactly anti-sunward")
    out["epoch_table"] = [
        dict(quantity="spacecraft initial state / propagation t=0",
             et=et_scen, source="phase8_X_ext at T_ext=0, propagated to phase7_et0 "
                                "(manifest x0_truth_provenance)"),
        dict(quantity="Sun position (SRP)", et="et0 + t",
             source="build_range_arc.sun_at: spkezr(SUN, et0 + t)"),
        dict(quantity="Earth rotation (ITRF93, sxform) for stations and visibility",
             et="pass_geo.et0_s + t = et0 + t_start + t_local",
             source="generate_two_way_range_measurements / sample_j2000_to_itrf93_transforms"),
        dict(quantity="Earth inertial position/velocity (station centre, light time)",
             et_legacy="campaign ET0 + t_local  (= et0 - 504000 s - t_start + ...)",
             et_canonical="pass_geo.et0_s + t_local",
             source="build_range_arc -> EarthEphemeris.pos/vel"),
        dict(quantity="range receive tag t3", et="pass_geo.et0_s + obs[:,0]",
             source="production two-way range; t1/t2u/t2d solved from t3 through the "
                    "same providers"),
        dict(quantity="dynamics Earth third body", et="n/a",
             source="mu_earth = 0; constant placeholder vector never used as a force"),
    ]
    return out


def clamp_characterization(arms: dict, recon: dict) -> dict:
    """s16: what 'Earth freezes after four orbits' means, measured."""
    import phase17_r1m_core as M
    from lunar_od.radiometrics import _interp_state

    C = M._legacy_campaign_module()  # noqa: N806
    t_obs_tab = np.asarray(C.T_OBS, float)
    t_orbit = recon["orbit_period_s"]
    a = arms["A_LEGACY_OFFSET_CLAMPED"]
    t = np.asarray(a["pass_t"])
    duration = float(W15_ORBITS * t_orbit)

    probe = np.array([-120.0, -60.0, 0.0, t_obs_tab[-1], t_obs_tab[-1] + 1.0,
                      t_obs_tab[-1] + 3000.0, t[-1]])
    p = np.asarray(C.get_earth_pos(probe))
    v = np.asarray(C.get_earth_vel(probe))
    held_after = bool(np.array_equal(p[3], p[4]) and np.array_equal(p[3], p[5])
                      and np.array_equal(p[3], p[6]))
    held_before = bool(np.array_equal(p[0], p[2]) and np.array_equal(p[1], p[2]))
    vel_after_norm = float(np.linalg.norm(v[6]))

    # production Hermite interpolation of the frozen history (event epochs fall
    # between grid nodes): constant position + non-zero velocity -> spurious wobble
    pos_hist = np.asarray(a["pass_earth_pos"])
    vel_hist = np.asarray(a["pass_earth_vel"])
    hist = np.hstack([pos_hist, vel_hist])
    k0 = int(np.searchsorted(t, t_obs_tab[-1] + 3 * CADENCE_S))
    s = np.linspace(t[k0], t[k0 + 1], 401)
    wob = max(float(np.linalg.norm(_interp_state(t, hist, x)[:3] - pos_hist[k0])) for x in s)

    obs_t = np.asarray(a["obs_t"])
    n_after = int(np.sum(obs_t > t_obs_tab[-1]))
    return dict(
        table_duration_s=float(t_obs_tab[-1] - t_obs_tab[0]),
        table_nodes=int(t_obs_tab.size), table_cadence_s=float(t_obs_tab[1] - t_obs_tab[0]),
        table_nominal_span="0 .. 4 * T_ORBIT (T_END), sampled with arange(0, T_END + 30, 60)",
        table_duration_orbits=float((t_obs_tab[-1] - t_obs_tab[0]) / t_orbit),
        arc_duration_s=duration, arc_grid_end_s=float(t[-1]),
        arc_duration_orbits=W15_ORBITS,
        clamp_start_scenario_t_s=float(t_obs_tab[-1]),
        clamp_start_et_table_epoch=float(C.ET0 + t_obs_tab[-1]),
        clamp_start_et_scenario_epoch=float(recon["scenario_et"] + t_obs_tab[-1]),
        clamp_start_orbit_number=float(t_obs_tab[-1] / t_orbit),
        interpolation_method="numpy.interp (piecewise linear) per component",
        out_of_range_behaviour_after_end="HELD_CONSTANT (last node), not extrapolated, not repeated",
        out_of_range_behaviour_before_start="HELD_CONSTANT (first node)",
        held_constant_after_end_verified=held_after,
        held_constant_before_start_verified=held_before,
        velocity_also_held_not_zeroed_mps=vel_after_norm,
        position_velocity_inconsistent_after_clamp=bool(vel_after_norm > 0.0),
        hermite_wobble_of_frozen_earth_between_nodes_m=wob,
        w15_legacy_obs_total=int(obs_t.size),
        w15_legacy_obs_after_clamp=n_after,
        w15_legacy_fraction_after_clamp=float(n_after / obs_t.size),
        arc_fraction_after_clamp=float((t[-1] - t_obs_tab[-1]) / t[-1]),
    )


def attribution(arms: dict) -> dict:
    f = {k: v["metrics"]["f_perp"] for k, v in arms.items()}
    s = {k: v["metrics"]["sigma_k_frac"] for k, v in arms.items()}
    A, B, Cc, D = (f[k] for k in ARMS)
    sA, sB, sC, sD = (s[k] for k in ARMS)
    return dict(
        total_repair_delta_f_perp=D - A,
        epoch_repair_given_no_clamp_delta_f_perp=D - B,
        epoch_repair_given_clamp_delta_f_perp=Cc - A,
        clamp_repair_given_correct_epoch_delta_f_perp=D - Cc,
        clamp_repair_given_offset_epoch_delta_f_perp=B - A,
        interaction_delta_f_perp=(D - B) - (Cc - A),
        epoch_main_effect_shapley_delta_f_perp=0.5 * ((D - B) + (Cc - A)),
        clamp_main_effect_shapley_delta_f_perp=0.5 * ((D - Cc) + (B - A)),
        total_repair_ratio_sigma_k_frac=sD / sA,
        epoch_repair_given_no_clamp_ratio_sigma_k_frac=sD / sB,
        clamp_repair_given_correct_epoch_ratio_sigma_k_frac=sD / sC,
    )


def main() -> None:
    _init()
    hdr("PHASE 17-FXR  s15 EPOCH-OFFSET RECONSTRUCTION")
    recon = epoch_reconstruction()
    for k, v in recon.items():
        print("  %-48s %s" % (k, v))
    offset_gate = (recon["campaign_module_et0_matches_mat"]
                   and recon["delta_matches_manifest_offset"]
                   and recon["manifest_canonical_et0"] == recon["historical_earth_table_start_et"])

    hdr("s17/s18  FOUR ARMS (A legacy, B offset-only, C clamp-only, D canonical)")
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        arms = {r["arm"]: r for r in ex.map(build_arm, list(ARMS))}
    print("  %-26s %5s %9s %8s %12s %9s %10s %6s %s" % (
        "arm", "n_obs", "f_perp", "theta_K", "I_K|x", "sigK/K", "||b_K||", "maxC", "bitwise-vs-GEO"))
    for k, a in arms.items():
        m = a["metrics"]
        print("  %-26s %5d %9.6f %8.4f %12.5e %8.4f%% %10.3f %6.3f %s" % (
            k, m["n_obs"], m["f_perp"], m["theta_k_deg"], m["conditional_k_information"],
            100 * m["sigma_k_frac"], m["k_column_norm"], m["max_abs_k_state_correlation"],
            a["bitwise_vs_geo_cache"]))
    same_truth = len({a["nom48_sha"] for a in arms.values()}) == 1
    print("  nominal trajectory (nom48) identical in all four arms: %s" % same_truth)

    # published digits (R1O / R1O-D / R1O-R / GEO all quote this baseline)
    pub = json.loads((ARTIFACTS / "r1od_campaign_summary.json").read_text())["range_only"]
    mA = arms["A_LEGACY_OFFSET_CLAMPED"]["metrics"]
    repro = dict(
        n_obs=(mA["n_obs"], pub["n_obs"]),
        f_perp=(mA["f_perp"], pub["orthogonal_fraction"]),
        theta_k_deg=(mA["theta_k_deg"], pub["theta_k_deg"]),
        conditional_k_information=(mA["conditional_k_information"],
                                   pub["conditional_k_information"]),
        sigma_k_frac=(mA["sigma_k_frac"], pub["fractional_sigma_k"]),
    )
    exact = all(a == b for a, b in repro.values())
    hist_gate = exact and arms["A_LEGACY_OFFSET_CLAMPED"]["bitwise_vs_geo_cache"]
    print("\n  legacy arm vs committed r1od_campaign_summary.json range_only: exact=%s" % exact)
    for k, (a, b) in repro.items():
        print("    %-26s reproduced %r   published %r" % (k, a, b))
    print("  HISTORICAL_FIXTURE_REPRODUCTION_GATE = %s" % ("PASS" if hist_gate else "FAIL"))

    hdr("s13/s15  RUNTIME EPOCH AUDIT")
    audit = runtime_audit(arms, recon)
    for k in ("A_LEGACY_OFFSET_CLAMPED", "D_CANONICAL"):
        print("  %s" % k)
        for kk, vv in audit[k].items():
            print("    %-62s %r" % (kk, vv))
    print("  Moon->Earth direction error of legacy vs canonical: %s"
          % audit["moon_to_earth_direction_error_deg"])
    print("  Sun-Moon-Earth angle: %s" % audit["sun_moon_earth_angle_deg"])
    canon = audit["D_CANONICAL"]
    legacy = audit["A_LEGACY_OFFSET_CLAMPED"]
    runtime_gate = (canon["earth_minus_rotation_origin_s"] == 0.0
                    and canon["max_abs_earth_pos_diff_vs_spice_at_measurement_et_m"] == 0.0
                    and legacy["earth_minus_rotation_origin_s"] == -recon["delta_et_seconds"])
    offset_gate = offset_gate and runtime_gate
    print("  EARTH_EPOCH_OFFSET_REPRODUCTION_GATE = %s" % ("PASS" if offset_gate else "FAIL"))
    print("  FXR_RUNTIME_EPOCH_AUDIT = %s" % ("PASS" if runtime_gate else "FAIL"))

    hdr("s16  EARTH-TABLE CLAMP CHARACTERIZATION")
    clamp = clamp_characterization(arms, recon)
    for k, v in clamp.items():
        print("  %-52s %s" % (k, v))
    clamp_gate = clamp["held_constant_after_end_verified"] and clamp["held_constant_before_start_verified"]
    print("  EARTH_TABLE_CLAMP_CHARACTERIZATION_GATE = %s" % ("PASS" if clamp_gate else "FAIL"))

    hdr("s17  ATTRIBUTION (changes caused by REPAIRING each defect)")
    attr = attribution(arms)
    for k, v in attr.items():
        print("  %-52s %+.6f" % (k, v))

    summary = dict(
        phase="17-FXR", configuration=dict(orbits=W15_ORBITS, stations=DSN3, cadence_s=CADENCE_S),
        epoch_reconstruction=recon, runtime_audit=audit, clamp=clamp, attribution=attr,
        arms={k: dict(fixture=a["fixture"], earth=a["earth"], n_obs=a["n_obs"],
                      bitwise_vs_geo_cache=a["bitwise_vs_geo_cache"], geo_cache=a["geo_cache"],
                      nom48_sha256=a["nom48_sha"], metrics=a["metrics"])
              for k, a in arms.items()},
        nominal_trajectory_identical_across_arms=same_truth,
        published_reproduction={k: dict(reproduced=a, published=b) for k, (a, b) in repro.items()},
        gates=dict(
            EARTH_EPOCH_OFFSET_REPRODUCTION_GATE="PASS" if offset_gate else "FAIL",
            FXR_RUNTIME_EPOCH_AUDIT="PASS" if runtime_gate else "FAIL",
            EARTH_TABLE_CLAMP_CHARACTERIZATION_GATE="PASS" if clamp_gate else "FAIL",
            HISTORICAL_FIXTURE_REPRODUCTION_GATE="PASS" if hist_gate else "FAIL",
        ),
    )
    (ARTIFACTS / "phase17_fxr_epoch_audit.json").write_text(
        json.dumps(summary, indent=1, default=float))
    print("\n  wrote artifacts/phase17_fxr_epoch_audit.json")


if __name__ == "__main__":
    main()
