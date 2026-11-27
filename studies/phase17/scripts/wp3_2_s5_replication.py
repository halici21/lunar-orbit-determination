"""F3 WP3.2 S5: predeclared hold-out replication of the WP3.1 A2 materiality classification (REQ-DYN-007).

Predeclaration: studies/phase17/configs/wp3_2_s5_replication_predeclaration.json. Population:
studies/phase17/results/wp3_2_s5_population.json (wp3_2_s5_population.py, committed before any survey result).

Each term is added alone to the production baseline (propagate_state, DOP853, shadow_events True = profile D2).
Production terms are switched on through propagate_state's own options; PN and PLANETS, which production lacks,
are WP3.1's study terms (wp3_1_force_terms.with_term) integrated with production's segmentation. The OD impact,
the response and the materiality rule are WP3.1's (wp3_1_materiality_survey: rows, design, fit).

    python wp3_2_s5_replication.py run H1 part_h1.json      # the three cells of one epoch
    python wp3_2_s5_replication.py combine part_h1.json ... part_h6.json
    python wp3_2_s5_replication.py smoke out.json           # 1 h arc, H3xS1, software check only
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import wp3_1_force_terms as W  # noqa: E402
import wp3_1_materiality_survey as S  # noqa: E402

PRE = ROOT / "studies" / "phase17" / "configs" / "wp3_2_s5_replication_predeclaration.json"
POPULATION = ROOT / "studies" / "phase17" / "results" / "wp3_2_s5_population.json"
OUT = ROOT / "studies" / "phase17" / "results" / "wp3_2_s5_replication.json"

PRODUCTION_TERMS = ("TIDE", "LRP", "EARTH_SHADOW", "SRP_LIGHT_TIME", "EARTH_J2")
STUDY_TERMS = ("PN", "PLANETS")
REPORTED = ("EARTH_J2_ITRF93",)
TERMS = PRODUCTION_TERMS + STUDY_TERMS
MARGINAL = ("EARTH_J2", "SRP_LIGHT_TIME")
A2_CLASS = {"TIDE": "IMPLEMENT", "LRP": "IMPLEMENT", "EARTH_SHADOW": "IMPLEMENT", "SRP_LIGHT_TIME": "IMPLEMENT",
            "EARTH_J2": "IMPLEMENT", "PN": "DOCUMENT_ONLY", "PLANETS": "DOCUMENT_ONLY"}
THRESHOLD = 0.1
MARGINAL_FLOOR = 0.01
FLOOR_MAX_M = 1e-3
G4_MAX, G4_INFO = 1e-3, 1e-4
G4_CELL = "H3xS1"
ECLIPSE_EPOCHS = ("H3", "H4", "H5", "H6")
EPOCHS = ("H1", "H2") + ECLIPSE_EPOCHS
G1_FILES = {"tests/test_wp3_2_s5_replication.py": TERMS,
            "tests/test_wp3_1_force_terms.py": ("PN", "PLANETS"),
            "tests/test_wp3_2_lunar_tide.py": ("TIDE",),
            "tests/test_wp3_2_lunar_radiation.py": ("LRP",),
            "tests/test_wp3_2_earth_shadow.py": ("EARTH_SHADOW",),
            "tests/test_wp3_2_srp_light_time.py": ("SRP_LIGHT_TIME",)}

S.PROFILE = "D2"          # every WP3.1 helper used here (design, G4 finite differences) runs event-segmented


# ======================================================================
# terms
# ======================================================================
def production_options(term: str, ssb_table, off: bool = False) -> dict:
    """propagate_state keyword arguments that switch one production term on (or explicitly off)."""
    from lunar_od.earth_shadow import EarthShadowOptions
    from lunar_od.constants import J2_EARTH_UNNORMALIZED
    from lunar_od.lunar_radiation import LunarRadiationOptions
    from lunar_od.lunar_tide import LunarTideOptions
    from lunar_od.srp_light_time import SRPLightTimeOptions
    if term == "EARTH_J2":
        return dict(j2_earth=0.0 if off else J2_EARTH_UNNORMALIZED, earth_j2_mode="indirect")
    key, make = {"TIDE": ("lunar_tide", lambda: LunarTideOptions(static_field_convention="TIDE_FREE")),
                 "LRP": ("lunar_radiation", LunarRadiationOptions),
                 "EARTH_SHADOW": ("earth_shadow", EarthShadowOptions),
                 "SRP_LIGHT_TIME": ("srp_light_time", lambda: SRPLightTimeOptions(ssb_velocities=ssb_table))}[term]
    return {key: None if off else make()}


def prod_run(cell, opts: dict, rtol: float = S.RTOL, atol: float = S.ATOL) -> np.ndarray:
    from lunar_od.dynamics import propagate_state
    return propagate_state(S.T_GRID, cell.x0, *cell.mus, cell.eph.earth_position, cell.eph.sun_position,
                           rtol=rtol, atol=atol, method="DOP853", srp=cell.srp, shadow_events=True, **opts)


def study_run(cell, term: str, scale: float, rtol: float = S.RTOL, atol: float = S.ATOL) -> np.ndarray:
    name = "EARTH_J2" if term == "EARTH_J2_ITRF93" else term
    rhs, sun = S.closure(cell)
    states, _ = S.integrate(W.with_term(rhs, name, S.context(cell, sun), scale), sun, cell.srp, cell.x0, rtol, atol)
    return states


def term_run(cell, term: str, ssb, rtol: float = S.RTOL, atol: float = S.ATOL) -> np.ndarray:
    if term in PRODUCTION_TERMS:
        return prod_run(cell, production_options(term, ssb), rtol, atol)
    return study_run(cell, term, 1.0, rtol, atol)


# ======================================================================
# cells
# ======================================================================
def build_cell(entry: dict, epoch: dict):
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters
    from lunar_od.srp import SRPOptions
    from lunar_od.srp_light_time import sample_ssb_velocities
    et0 = float(epoch["et0"])
    t_eph = np.arange(-S.EPH_MARGIN_S, float(S.T_GRID[-1]) + S.EPH_MARGIN_S + 0.5, S.EPH_STEP_S)
    eph = sample_moon_centered_ephemeris(et0, t_eph)
    info = {k: entry[k] for k in ("stratum", "beta_deg", "omega_rad", "u_rad", "altitude_m",
                                  "inclination_to_lunar_equator_deg")}
    cell = S.Cell(entry["name"], entry["epoch"], et0, np.asarray(entry["x0"], dtype=float), eph,
                  gravitational_parameters("DE440"), SRPOptions(k_srp_m2_per_kg=W.K_SRP_M2_PER_KG), W.planet_gm(),
                  dict(info, kind=epoch["kind"], t0_utc=epoch["t0_utc"]))
    return cell, sample_ssb_velocities(et0, t_eph)


def _max_dr(a, b) -> float:
    return float(np.max(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)))


def run_cell(cell, ssb, do_g4: bool) -> dict:
    t_start = time.time()
    zero = np.zeros(3)
    x_base = prod_run(cell, {})
    floor = _max_dr(prod_run(cell, {}, *S.FLOOR_TOL), x_base)
    obs, pass_geo, vis = S.measurement_rows(cell, x_base)
    h, nom = S.design(cell, x_base, obs, pass_geo)
    y_base = S.observables(x_base, obs, pass_geo)
    sun, earth = cell.eph.sun_position(S.T_GRID), cell.eph.earth_position(S.T_GRID)
    nu_moon = np.array([W.lunar_illumination(x[:3], s, zero, cell.srp) for x, s in zip(x_base, sun)])
    nu_e_sc = np.array([W.earth_illumination(x[:3], s, e) for x, s, e in zip(x_base, sun, earth)])
    nu_e_moon = np.array([W.earth_illumination(zero, s, e) for s, e in zip(sun, earth)])
    nom48 = _max_dr(nom, x_base)
    out = dict(cell=cell.name, epoch=cell.epoch, et0=cell.et0, x0=cell.x0.tolist(), info=cell.info,
               floor_m=floor, nom48_vs_baseline_m=nom48,
               G3=bool(floor <= FLOOR_MAX_M and nom48 <= FLOOR_MAX_M),
               rows=int(obs.shape[0]), rows_per_station=np.sum(vis, axis=0).tolist(),
               lunar_shadow=dict(fraction_nu_below_1=float(np.mean(nu_moon < 1.0)),
                                 fraction_umbra=float(np.mean(nu_moon == 0.0))),
               earth_shadow_sc=dict(fraction_nu_below_1=float(np.mean(nu_e_sc < 1.0)), min_nu=float(nu_e_sc.min())),
               earth_shadow_moon_centre=dict(min_nu=float(nu_e_moon.min()),
                                             any_umbra=bool(np.any(nu_e_moon == 0.0)),
                                             all_lit=bool(np.all(nu_e_moon == 1.0)),
                                             epochs_below_1=int(np.sum(nu_e_moon < 1.0))))
    print(f"[{cell.name}] floor {floor:.3e} m, nom48 {nom48:.3e} m, rows {obs.shape[0]}", flush=True)
    terms = {}
    for term in TERMS + REPORTED:
        t_term = time.time()
        x_term = term_run(cell, term, ssb)
        x_term_fine = term_run(cell, term, ssb, *S.FLOOR_TOL)
        if term in PRODUCTION_TERMS:
            x_null = prod_run(cell, production_options(term, ssb, off=True))
        else:
            x_null = study_run(cell, term, 0.0)
        dr = np.linalg.norm(x_term[:, :3] - x_base[:, :3], axis=1)
        dy = S.observables(x_term, obs, pass_geo) - y_base
        fits = {f"{d}/{n}": S.fit(h, dy, *S.NOISE[n], d == "RANGE_RANGE_RATE") for d in S.DESIGNS for n in S.NOISE}
        best = max(fits, key=lambda k: fits[k]["response"])
        term_floor = _max_dr(x_term_fine, x_term)
        terms[term] = dict(source="production" if term in PRODUCTION_TERMS else "study",
                           G2=bool(np.array_equal(x_null, x_base)), max_dr_m=float(dr.max()),
                           final_dr_m=float(dr[-1]), floor_m=term_floor,
                           below_floor=bool(dr.max() < 10.0 * (floor + term_floor)),
                           max_dy_range_m=float(np.max(np.abs(dy[:, 0]))),
                           max_dy_range_rate_mps=float(np.max(np.abs(dy[:, 1]))),
                           response=fits[best]["response"], response_at=best, fits=fits,
                           wall_s=time.time() - t_term)
        print(f"[{cell.name}] {term}: max|dr| {dr.max():.3e} m, floor {term_floor:.2e}, response "
              f"{fits[best]['response']:.3e} ({best}), G2 {terms[term]['G2']}, {time.time() - t_term:.0f} s",
              flush=True)
    out["terms"] = terms
    if do_g4:
        g = S.g5_design_check(cell, obs, pass_geo, h)
        out["G4"] = dict(relative=g["relative"], worst=g["worst"], PASS=bool(g["worst"] <= G4_MAX),
                         info_1e_4=bool(g["worst"] <= G4_INFO), steps=g["steps"])
        print(f"[{cell.name}] G4 worst {g['worst']:.3e}", flush=True)
    out["wall_s"] = time.time() - t_start
    return out


def run(epoch: str, out_path: str, names=None, smoke: bool = False) -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    pop = json.loads(POPULATION.read_text(encoding="utf-8"))
    cells = []
    for entry in pop["cells"]:
        if entry["epoch"] != epoch or (names and entry["name"] not in names):
            continue
        cell, ssb = build_cell(entry, pop["epochs"][epoch])
        cells.append(run_cell(cell, ssb, do_g4=(cell.name == G4_CELL and not smoke)))
        Path(out_path).write_text(json.dumps(dict(epoch=epoch, smoke=smoke, arc_s=float(S.T_GRID[-1]),
                                                  cells=cells), indent=1), encoding="utf-8")


# ======================================================================
# decisions (pure)
# ======================================================================
def classify(term: str, cells: list) -> dict:
    rows = []
    for c in cells:
        if term == "EARTH_SHADOW" and c["epoch"] not in ECLIPSE_EPOCHS:
            continue
        t = c["terms"][term]
        floors = c["floor_m"] + t["floor_m"]
        rows.append(dict(cell=c["cell"], response=t["response"], at=t["response_at"], max_dr_m=t["max_dr_m"],
                         usable=bool(t["max_dr_m"] >= 10.0 * floors),
                         inconclusive_in_cell=bool((not c["G3"]) and t["max_dr_m"] < 100.0 * floors)))
    usable = [r for r in rows if r["usable"] and not r["inconclusive_in_cell"]]
    top = max(usable, key=lambda r: r["response"]) if usable else None
    max_usable = top["response"] if top else 0.0
    material = max_usable >= THRESHOLD
    any_inconclusive = any(r["inconclusive_in_cell"] for r in rows)
    if material:
        cls = "IMPLEMENT"
    elif any_inconclusive:
        cls = "S5_INCONCLUSIVE"
    else:
        cls = "DOCUMENT_ONLY"
    top_all = max(rows, key=lambda r: r["response"]) if rows else None
    return dict(hold_out_class=cls, OD_MATERIAL=material, max_usable_response=max_usable,
                at=f"{top['cell']} {top['at']}" if top else None,
                max_response_any_cell=top_all["response"] if top_all else None,
                max_dr_m=max((r["max_dr_m"] for r in rows), default=0.0),
                cells_below_floor=[r["cell"] for r in rows if not r["usable"]], cells=rows)


def replication(term: str, hold_out_class: str, max_response: float) -> str:
    if hold_out_class == "S5_INCONCLUSIVE":
        return "S5_INCONCLUSIVE"
    if term in MARGINAL:
        if max_response >= THRESHOLD:
            return "REPLICATED"
        return "MARGINAL_NOT_REPLICATED" if max_response >= MARGINAL_FLOOR else "CONTRADICTED"
    return "REPLICATED" if hold_out_class == A2_CLASS[term] else "NOT_REPLICATED"


def req_dyn_007(verdicts: dict, gates_pass: bool) -> str:
    if not gates_pass:
        return "partial"
    strict = all(verdicts[t] == "REPLICATED" for t in TERMS if t not in MARGINAL)
    marginal = all(verdicts[t] in ("REPLICATED", "MARGINAL_NOT_REPLICATED") for t in MARGINAL)
    return "passed" if strict and marginal else "partial"


# ======================================================================
# gates and combine
# ======================================================================
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def g1_unit() -> dict:
    per_file = {}
    for f in G1_FILES:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", f],
                              cwd=ROOT, capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
        per_file[f] = dict(PASS=proc.returncode == 0, summary=lines[-1] if lines else proc.stdout[-300:])
    per_term = {t: all(per_file[f]["PASS"] for f, ts in G1_FILES.items() if t in ts) for t in TERMS}
    return dict(PASS=all(v["PASS"] for v in per_file.values()), per_file=per_file, per_term=per_term,
                s5_file_pass=per_file["tests/test_wp3_2_s5_replication.py"]["PASS"])


def decide(cells: list, g1: dict, pop: dict) -> dict:
    g2a = {t: all(c["terms"][t]["G2"] for c in cells) for t in PRODUCTION_TERMS}
    g2b = all(c["terms"][t]["G2"] for c in cells for t in STUDY_TERMS + REPORTED)
    g3 = {c["cell"]: c["G3"] for c in cells}
    g4_cell = [c for c in cells if c["cell"] == G4_CELL]
    g4 = bool(g4_cell and g4_cell[0]["G4"]["PASS"])
    by_epoch = {e: [c for c in cells if c["epoch"] == e] for e in EPOCHS}
    g5 = {}
    for e in ("H3", "H4"):
        g5[f"{e}_moon_centre_umbra"] = any(c["earth_shadow_moon_centre"]["any_umbra"] for c in by_epoch[e])
    for e in ("H5", "H6"):
        mins = [c["earth_shadow_moon_centre"]["min_nu"] for c in by_epoch[e]]
        g5[f"{e}_moon_centre_partial"] = bool(mins and 0.0 < min(mins) < 1.0)
    for e in ECLIPSE_EPOCHS:
        g5[f"{e}_earth_shadow_nonzero"] = any(c["terms"]["EARTH_SHADOW"]["max_dr_m"] > 0.0 for c in by_epoch[e])
    for e in ("H1", "H2"):
        g5[f"{e}_all_lit"] = all(c["earth_shadow_moon_centre"]["all_lit"] and c["earth_shadow_sc"]["min_nu"] == 1.0
                                 for c in by_epoch[e])
        g5[f"{e}_earth_shadow_zero"] = all(c["terms"]["EARTH_SHADOW"]["max_dr_m"] == 0.0 for c in by_epoch[e])
    g5_pass = all(g5.values())
    g6 = pop["S5_G6_INDEPENDENCE"]
    common = g1["s5_file_pass"] and g2b and g4 and g6["PASS"]
    classes, verdicts = {}, {}
    for term in TERMS:
        cls = classify(term, cells)
        veto = (not common) or (not g1["per_term"][term]) or (term in g2a and not g2a[term]) or \
               (term == "EARTH_SHADOW" and not g5_pass)
        if veto:
            cls["hold_out_class"] = "S5_INCONCLUSIVE"
        classes[term] = cls
        verdicts[term] = replication(term, cls["hold_out_class"], cls["max_usable_response"])
    gates = dict(S5_G1_UNIT=g1["PASS"], S5_G2_NULL=dict(production=g2a, study_PASS=g2b),
                 S5_G3_FLOOR=g3, S5_G4_DESIGN=g4, S5_G5_ECLIPSE=g5_pass, S5_G6_INDEPENDENCE=g6["PASS"])
    gates_pass = bool(g1["PASS"] and all(g2a.values()) and g2b and all(g3.values()) and g4 and g5_pass
                      and g6["PASS"])
    reported = {k: dict(max_response=max(c["terms"][k]["response"] for c in cells),
                        max_dr_m=max(c["terms"][k]["max_dr_m"] for c in cells)) for k in REPORTED}
    return dict(gates=gates, gates_all_pass=gates_pass, G1=g1, G4=g4_cell[0]["G4"] if g4_cell else None, G5=g5,
                G6=g6, classes=classes, replication=verdicts, REQ_DYN_007=req_dyn_007(verdicts, gates_pass),
                reported=reported)


def combine(parts: list) -> None:
    import scipy
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in parts]
    if any(p.get("smoke") for p in loaded):
        raise ValueError("a smoke part cannot enter the result")
    cells = [c for p in loaded for c in p["cells"]]
    pop = json.loads(POPULATION.read_text(encoding="utf-8"))
    if sorted(c["cell"] for c in cells) != sorted(c["name"] for c in pop["cells"]):
        raise ValueError("the 18 population cells are required")
    g1 = g1_unit()
    result = dict(predeclaration=str(PRE.relative_to(ROOT)).replace("\\", "/"), predeclaration_sha256=_sha256(PRE),
                  population=str(POPULATION.relative_to(ROOT)).replace("\\", "/"),
                  population_sha256=_sha256(POPULATION), status="as predeclared",
                  commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  threshold=THRESHOLD, marginal_floor=MARGINAL_FLOOR, a2_class=A2_CLASS, noise=S.NOISE,
                  prior_sd=S.PRIOR_SD.tolist(), parameters=S.PARAMS, **decide(cells, g1, pop), cells=cells)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gates", "gates_all_pass", "replication", "REQ_DYN_007")}, indent=1))


def main(argv: list) -> None:
    mode = argv[1]
    if mode == "run":
        run(argv[2], argv[3])
    elif mode == "smoke":
        S.T_GRID = np.arange(0.0, 3600.0 + 30.0, 60.0)
        run("H3", argv[2], names=(G4_CELL,), smoke=True)
    elif mode == "combine":
        combine(argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
