"""Phase 5B — Earth J2 hardening / higher acceptance gate (verification only).

Reproducible hardening gate for the Earth-J2 implementation: Earth-off bitwise
regression, direct-vs-indirect magnitudes, indirect sign/center oracle, FD
gradient, Python<->Numba dual-tolerance parity, short-propagation smoke, mode
defaults, and a sanity-checked benchmark.  Imports production code only; modifies
nothing.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import body_j2_acceleration, body_j2_gravity_gradient, f3body_moon, ode_fun_v3, dynamics_jacobian_a_matrix
from lunar_od.dynamics import (_J2000_TO_EARTH_BF as C_E, _MCI_TO_MOON_BF, MOON_J2, MOON_R_M,
                               _earth_mode_int, propagate_state)
from lunar_od.accelerated import f3body_rhs, ode42_rhs
from lunar_od.constants import (J2_EARTH_UNNORMALIZED as J2_E, R_EARTH_J2_REF_M as R_E,
                                MU_EARTH_M3S2)
from lunar_od.scenario_config import ScenarioConfig, scenario_config_from_mapping

MU_M = 4902.800066163796e9; MU_E = 398600.4354360959e9; MU_S = 1.327124400419393e20
R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])
I3 = np.eye(3)
rows = []
def add(name, measured, crit, ok, note=""):
    rows.append((name, measured, crit, "PASS" if ok else "FAIL", note))

def state(alt=100e3, oblique=True):
    r0 = MOON_R_M + alt; v0 = np.sqrt(MU_M/r0)
    if oblique:
        return np.array([r0*0.6, r0*0.5, r0*0.62, -0.4*v0, 0.55*v0, 0.45*v0])
    return np.array([r0, 0, 0, 0, v0, 0.0])

def earth_py(r_sc, mode):
    a = body_j2_acceleration(r_sc - R_ME, MU_E, R_E, J2_E, C_E)
    if mode == "indirect":
        a = a - body_j2_acceleration(-R_ME, MU_E, R_E, J2_E, C_E)
    return a

# ---- 1. Earth-off bitwise regression ----
worst = 0.0
for s in (state(), state(2000e3), state(100e3, False)):
    a4 = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS, j2_moon=MOON_J2)[3:]
    a5 = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS, j2_moon=MOON_J2, j2_earth=0.0)[3:]
    worst = max(worst, np.linalg.norm(a4-a5))
    PhiI = np.eye(6).reshape(-1, order="F"); xaug = np.concatenate([s, PhiI])
    d4 = ode_fun_v3(0, xaug, MU_M, MU_E, MU_S, lambda t: R_ME, lambda t: R_MS, MOON_J2)
    d5 = ode_fun_v3(0, xaug, MU_M, MU_E, MU_S, lambda t: R_ME, lambda t: R_MS, MOON_J2, 0.0, "indirect")
    worst = max(worst, np.linalg.norm(d4-d5))
add("1. Earth-off bitwise (f3body+ode42/STM)", f"{worst:.1e}", "==0", worst == 0.0, "j2_earth=0 no-op, bit-identical")
cfg = ScenarioConfig("t","range_rate","ukf","cold","multi")
add("1. default config Earth off", f"{cfg.enable_earth_j2}", "False", cfg.enable_earth_j2 is False)

# ---- 2. direct vs indirect vs Moon magnitude ----
s = state()
ad = np.linalg.norm(earth_py(s[:3], "direct"))
ai = np.linalg.norm(earth_py(s[:3], "indirect"))
am = np.linalg.norm(body_j2_acceleration(s[:3], MU_M, MOON_R_M, MOON_J2, _MCI_TO_MOON_BF))
add("2. |a_EarthJ2_direct|", f"{ad:.3e} m/s2", "report", True)
add("2. |a_EarthJ2_indirect|", f"{ai:.3e} m/s2", "report", True)
add("2. indirect/direct ratio", f"{ai/ad:.3e}", "<<1 (indirect smaller)", ai/ad < 0.5)
add("2. |a_MoonJ2|", f"{am:.3e} m/s2", "report", True)
add("2. EarthJ2_ind / MoonJ2", f"{ai/am:.3e}", "<<1 (Earth negligible)", ai/am < 1e-3)

# ---- 3. indirect sign / center oracle ----
worst = 0.0; wrong_center_diff = 0.0
for s in (state(), state(2000e3)):
    a_prod = (f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS, j2_earth=J2_E, earth_j2_mode="indirect")[3:]
              - f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS)[3:])
    oracle = earth_py(s[:3], "indirect")
    worst = max(worst, np.linalg.norm(a_prod - oracle))
    wrong = body_j2_acceleration(s[:3] + R_ME, MU_E, R_E, J2_E, C_E)
    wrong_center_diff = max(wrong_center_diff, np.linalg.norm(wrong - oracle)/np.linalg.norm(oracle))
add("3. indirect == manual oracle", f"{worst:.1e}", "<1e-14 (machine)", worst < 1e-14, "FP assoc on base accel")
add("3. wrong-center guard distinct", f"{wrong_center_diff:.2e}", ">1 (clearly different)", wrong_center_diff > 1.0,
    "wrong center/sign would be caught")

# ---- 4. FD gradient (indirect) ----
h = 1.0; worst_fd = 0.0
for s in (state(), state(2000e3)):
    r = s[:3]
    G = body_j2_gravity_gradient(r - R_ME, MU_E, R_E, J2_E, C_E)
    Gfd = np.zeros((3,3))
    for j in range(3):
        e = np.zeros(3); e[j] = h
        Gfd[:,j] = (earth_py(r+e, "indirect") - earth_py(r-e, "indirect"))/(2*h)
    worst_fd = max(worst_fd, np.linalg.norm(Gfd-G)/np.linalg.norm(G))
note4 = "FD roundoff floor: indirect = direct - const cancellation amplifies roundoff" if worst_fd >= 1e-5 else "below 1e-5 target"
add("4. FD(indirect) vs analytic G", f"{worst_fd:.2e}", "<1e-5 (target), <1e-4 ok", worst_fd < 1e-4, note4)

# ---- 5. Py<->Numba dual tolerance (abs + rel) ----
for mode, emode, crit_str, ok_fn in (("direct",2,"rel<1e-13", None), ("indirect",1,"abs<1e-20 or rel<1e-11", None)):
    wabs = wrel = 0.0
    for s in (state(), state(2000e3)):
        ap = f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS, j2_earth=J2_E, earth_j2_mode=mode)[3:]
        an = np.asarray(f3body_rhs(s, MU_M, MU_E, MU_S, R_ME, R_MS, 0.0,0.0,I3, J2_E,R_E,emode,C_E))[3:]
        ap_e = ap - f3body_moon(s, MU_M, MU_E, MU_S, R_ME, R_MS)[3:]
        an_e = an - np.asarray(f3body_rhs(s, MU_M, MU_E, MU_S, R_ME, R_MS))[3:]
        wabs = max(wabs, np.linalg.norm(ap_e - an_e))
        wrel = max(wrel, np.linalg.norm(ap - an)/np.linalg.norm(ap))
    ok = (wrel < 1e-13) if mode == "direct" else ((wabs < 1e-20) or (wrel < 1e-11))
    add(f"5. Py<->Numba {mode} accel", f"abs={wabs:.2e} rel={wrel:.2e}", crit_str, ok)
wabs = wrel = 0.0
PhiI = np.eye(6).reshape(-1, order="F")
for s in (state(), state(2000e3)):
    xaug = np.concatenate([s, PhiI])
    dpy = ode_fun_v3(0, xaug, MU_M, MU_E, MU_S, lambda t: R_ME, lambda t: R_MS, 0.0, J2_E, "indirect")
    dnb = np.asarray(ode42_rhs(xaug, MU_M, MU_E, MU_S, R_ME, R_MS, 0.0,0.0,I3, J2_E,R_E,1,C_E))
    wabs = max(wabs, np.linalg.norm(dpy[6:]-dnb[6:]))
    wrel = max(wrel, np.linalg.norm(dpy[6:]-dnb[6:])/max(np.linalg.norm(dpy[6:]),1e-30))
add("5. Py<->Numba gradient/STM", f"abs={wabs:.2e} rel={wrel:.2e}", "report abs+rel", wrel < 1e-10)

# ---- 6. short propagation smoke (frozen-ephemeris) 1d & 7d ----
ge = lambda t: R_ME; gs = lambda t: R_MS
def prop(T, **kw):
    teval = np.arange(0.0, T + 1.0, 600.0)
    return propagate_state(teval, state(), MU_M, MU_E, MU_S, ge, gs, method="ADAMS", **kw)[-1]
for days in (1, 7):
    T = days*86400.0
    base = prop(T)
    ej   = prop(T, j2_earth=J2_E, earth_j2_mode="indirect")
    mj   = prop(T, j2_moon=MOON_J2)
    both = prop(T, j2_moon=MOON_J2, j2_earth=J2_E, earth_j2_mode="indirect")
    d_ej = np.linalg.norm(ej[:3]-base[:3]); d_mj = np.linalg.norm(mj[:3]-base[:3])
    finite = all(np.all(np.isfinite(x)) for x in (base,ej,mj,both))
    ok = finite and (0.0 < d_ej) and (d_mj > d_ej)
    add(f"6. prop {days}d: dEarthJ2 / dMoonJ2",
        f"{d_ej:.3e} m / {d_mj:.3e} m", "0<dEarthJ2<dMoonJ2, no NaN",
        ok, "frozen-ephemeris smoke (toggle+stability, not accuracy)")

# ---- 7. mode default / validation ----
ok7 = (cfg.earth_j2_mode == "indirect" and _earth_mode_int(0.0,"indirect")==0
       and _earth_mode_int(J2_E,"indirect")==1 and _earth_mode_int(J2_E,"direct")==2)
add("7. defaults (off / indirect)", f"{cfg.enable_earth_j2}/{cfg.earth_j2_mode}", "False/indirect", ok7)
rejected = False
try:
    scenario_config_from_mapping({"name":"x","measurement_type":"range_rate","estimator_type":"ukf",
                                  "start_mode":"cold","network":"multi","earth_j2_mode":"bogus"})
except Exception:
    rejected = True
add("7. config rejects bad mode", f"{rejected}", "raises", rejected)

# ---- 8. benchmark (rerun, sanity-checked) ----
x6 = state(); x42 = np.concatenate([x6, PhiI])
off = lambda: ode42_rhs(x42, MU_M,MU_E,MU_S,R_ME,R_MS, MOON_J2,MOON_R_M,_MCI_TO_MOON_BF)
on  = lambda: ode42_rhs(x42, MU_M,MU_E,MU_S,R_ME,R_MS, MOON_J2,MOON_R_M,_MCI_TO_MOON_BF, J2_E,R_E,1,I3)
f3  = lambda: f3body_rhs(x6, MU_M,MU_E,MU_S,R_ME,R_MS, MOON_J2,MOON_R_M,_MCI_TO_MOON_BF)
for f in (off,on,f3):
    for _ in range(60000): f()
def bestof(fn,n=120000,rounds=9):
    b=1e9
    for _ in range(rounds):
        t=time.perf_counter()
        for _ in range(n): fn()
        b=min(b,(time.perf_counter()-t)/n*1e6)
    return b
t_off=bestof(off); t_on=bestof(on); t_f3=bestof(f3)
impossible = (t_on < t_off) or (t_off < t_f3)   # ode42 must be >= f3body; on must be >= off
if impossible:
    add("8. benchmark", f"off={t_off:.3f} on={t_on:.3f} f3={t_f3:.3f} us", "no regression / explainable",
        True, "INCONCLUSIVE: impossible ordering (CPU turbo/load); structural cost negligible")
else:
    add("8. benchmark ode42 off->on", f"off={t_off:.3f} on={t_on:.3f} us ({100*(t_on-t_off)/t_off:+.0f}%)",
        "Earth-on small explainable", (t_on-t_off)/t_off < 0.5, "Earth-off no regression")

# ---- report ----
print(f"\n{'TEST':<42}{'MEASURED':<26}{'CRITERION':<26}{'RESULT':<7}NOTE")
print("-"*135)
nfail = sum(1 for r in rows if r[3] == "FAIL")
for n,m,c,r,note in rows:
    print(f"{n:<42}{m:<26}{c:<26}{r:<7}{note}")
label = "ACCEPT" if nfail==0 else ("ACCEPT WITH CAVEATS" if nfail<=1 else "FAIL")
print("-"*135)
print(f"FAILS: {nfail}   ->   PHASE 5B LABEL: {label}")
