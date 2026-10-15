"""PHASE 17 model-mismatch pivot: linear probe of lunar planetary radiation pressure (albedo + thermal IR).

A probe, not a campaign: it decides whether the candidate next science target leaves a signature that range +
counted Doppler can see, and how much of it an SRP-only model would absorb as K / K variation.

Force (declared, deliberately simple; Knocke, Ries & Tapley 1988 style, sub-spacecraft-point approximation of a
uniform Lambertian sphere): outward radial

    a_PR = K_SRP * (R_M / r)^2 * [ A_b S(d) max(cos z, 0) + max(sigma T_ss^4 cos z, sigma T_n^4) ] / c * r_hat

z = solar zenith angle of the sub-spacecraft point, A_b = 0.12 (lunar Bond albedo), T_ss = 390 K (subsolar),
T_n = 95 K (night), S(d) the solar irradiance at the Moon-Sun distance; the same cannonball K as SRP (same area
and reflectivity).  A dimensionless scale c_PR multiplies it (truth 1 = fully present).

Trajectory sensitivity to c_PR by STM convolution, S_PR(t) = Phi(t) int_0^t Phi(tau)^-1 [0; a_PR(tau)] dtau
(trapezoid on the measurement grid), checked by applying the SAME construction to the production SRP kernel and
comparing with the production dx/dK_SRP.  Measurement columns through the KP2/KGM production evaluation.

Metrics per cell, on the KP2 estimator's design (x0 with its a priori, M3 K segments, one free bias per Doppler
pass; range 5 m, Doppler 1e-4 m/s):
  sine_vs_M3      ||(I-P_A) h|| / ||h|| of the whitened PR column against that design (separability)
  sine_K_vs_state the same for the scalar SRP K column against x0 + biases (the KP1/CD1 reference f_perp)
  resid_chi2_10   chi^2 of the unabsorbed PR signature for a 10 % PR mismodel (detectability)
  alias_10        what a 10 % PR mismodel does to an SRP-only M3 solve: false-variation chi^2 (2 dof, vs 13.82),
                  common-level bias / sigma, state bias
  sigma_c         formal sigma of c_PR if it is estimated alongside the M3 design
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kgm_core as G  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

K = P.K
OUT = K2.ARTIFACTS / "phase17_mm_probe.json"
A_BOND, T_SS, T_NIGHT = 0.12, 390.0, 95.0
SIGMA_SB = 5.670374419e-8


def pr_acceleration(r, r_sun, k_srp):
    from lunar_od.constants import AU_M, R_MOON_M, SOLAR_IRRADIANCE_1AU_W_M2, SPEED_OF_LIGHT_M_S
    rn = float(np.linalg.norm(r))
    r_hat = r / rn
    d = float(np.linalg.norm(r_sun))
    cz = float(r_hat @ (r_sun / d))
    s = SOLAR_IRRADIANCE_1AU_W_M2 * (AU_M / d) ** 2
    exitance = A_BOND * s * max(cz, 0.0) + max(SIGMA_SB * T_SS ** 4 * cz, SIGMA_SB * T_NIGHT ** 4)
    return k_srp * (R_MOON_M / rn) ** 2 * exitance / SPEED_OF_LIGHT_M_S * r_hat


def gravity_omission(scn, X, t, n_lo=50, n_hi=100, field_diff=False):
    """Acceleration of the GL1800F degrees n_lo+1..n_hi omitted by the n_lo x n_lo estimation field; with
    ``field_diff`` instead GRGM1200L(n_hi) - GL1800F(n_hi): two independent GRAIL solutions at the same degree,
    a commission-error proxy."""
    import od_gravity_covariance_campaign as C  # noqa: N811
    import spiceypy as spice
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration
    import dataclasses

    from lunar_od.gravity_model_loader import load_lunar_gravity_model

    def model(n):
        return C._M[n] if n in C._M else dataclasses.replace(
            load_lunar_gravity_model(C.GL1800F, nmax=n, mmax=n), frame=C.FRAME)
    m_hi, m_lo = model(n_hi), model(n_lo)
    if field_diff:
        m_hi = dataclasses.replace(load_lunar_gravity_model(
            C.GL1800F.replace("gl1800f/jggrx_1800f_sha.tab", "grgm1200l/gggrx_1200l_sha.tab"),
            nmax=n_hi, mmax=n_hi), frame=C.FRAME)
        m_lo = model(n_hi)
    out = np.zeros((t.size, 3))
    for i, ti in enumerate(t):
        c = np.asarray(spice.pxform("J2000", C.FRAME, scn.et0 + float(ti)))
        out[i] = (spherical_harmonic_acceleration(X[i, :3], m_hi, c)
                  - spherical_harmonic_acceleration(X[i, :3], m_lo, c))
    return out


def convolve(t, PHI, acc):
    """S(t) = Phi(t) int Phi^-1 B a, trapezoid on the grid."""
    integrand = np.stack([np.linalg.solve(PHI[i], np.concatenate([np.zeros(3), acc[i]])) for i in range(t.size)])
    psi = np.zeros_like(integrand)
    dt = np.diff(t)[:, None]
    psi[1:] = np.cumsum(0.5 * dt * (integrand[1:] + integrand[:-1]), axis=0)
    return np.einsum("nij,nj->ni", PHI, psi)


def _project_out(a, h):
    q, _ = np.linalg.qr(a)
    return h - q @ (q.T @ h)


def cell(spec):
    import phase17_geo_core as GC
    from lunar_od.srp import SRPOptions, srp_acceleration_kernel
    K._init()
    orbit, k, dyn = spec["orbit"], spec["k"], spec["dyn"]
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), dyn)
    t = scn.t_grid
    sun = np.stack([GC.sun_mci(scn.et0 + float(ti)) for ti in t])
    force = spec.get("force", "PR")
    if force == "GRAV":
        a_pr = gravity_omission(scn, X, t, *spec.get("degrees", (50, 100)), field_diff=spec.get("field_diff", False))
    else:
        a_pr = np.stack([pr_acceleration(X[i, :3], sun[i], K2.K0) for i in range(t.size)])
    opts = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    kern = np.stack([srp_acceleration_kernel(X[i, :3], sun[i], opts) for i in range(t.size)])
    s_pr = convolve(t, PHI, a_pr)
    s_k_conv = convolve(t, PHI, kern)
    s_k = S.sum(axis=0)
    conv_check = float(np.max(np.linalg.norm(s_k_conv[:, :3] - s_k[:, :3], axis=1)) /
                       np.max(np.linalg.norm(s_k[:, :3], axis=1)))
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, K.stations())
    ev = G.evaluate(fx, X, PHI, np.concatenate([S, s_pr[None], s_k[None]], axis=0))
    wr, wd = 1 / K2.SIGMA_RANGE_M, 1 / K2.SIGMA_DOPPLER_MPS
    q = fx.nuis.shape[1]
    hx = np.vstack([ev["hx_r"] * wr, ev["hx_d"] * wd])
    hk3 = np.vstack([ev["hk_r"][:, :3] * wr, ev["hk_d"][:, :3] * wd])
    h_pr = np.concatenate([ev["hk_r"][:, 3] * wr, ev["hk_d"][:, 3] * wd])
    h_k = np.concatenate([ev["hk_r"][:, 4] * wr, ev["hk_d"][:, 4] * wd])
    nu = np.vstack([np.zeros((ev["hx_r"].shape[0], q)), fx.nuis * wd])
    prior = np.zeros((6, 6 + 3 + q))
    prior[:, :6] = np.diag(1 / K2.PRIOR_SD)

    def design(blocks):
        a = np.hstack(blocks)
        pr = np.zeros((6, a.shape[1]))
        pr[:, :6] = np.diag(1 / K2.PRIOR_SD)
        return np.vstack([a, pr])

    pad = lambda v: np.concatenate([v, np.zeros(6)])  # noqa: E731
    a_m3 = design([hx, hk3, nu])
    a_st = design([hx, nu])
    cs = np.linalg.norm(a_m3, axis=0)
    an = a_m3 / cs
    hp = pad(h_pr)
    sine_pr = float(np.linalg.norm(_project_out(an, hp)) / np.linalg.norm(hp))
    cs_st = np.linalg.norm(a_st, axis=0)
    sine_k = float(np.linalg.norm(_project_out(a_st / cs_st, pad(h_k))) / np.linalg.norm(h_k))
    # alias of a 10 % PR mismodel into the SRP-only M3 solve
    eps = 0.10 if force == "PR" else 1.0      # PR: a 10 % mismodel; GRAV: the full omission (degrees 51-100)
    sol, *_ = np.linalg.lstsq(an, eps * hp, rcond=None)
    dtheta = sol / cs
    r = np.linalg.qr(an, mode="r")
    rinv = np.linalg.solve(r, np.eye(r.shape[0]))
    cov = (rinv @ rinv.T) / np.outer(cs, cs)
    df = dtheta[6:9] / K2.K0
    cf = cov[6:9, 6:9] / K2.K0 ** 2
    v = K2.Q_VAR.T @ df
    cv = K2.Q_VAR.T @ cf @ K2.Q_VAR
    one = np.ones(3) / 3
    a_c = np.hstack([an, (hp / np.linalg.norm(hp))[:, None]])
    rc = np.linalg.qr(a_c, mode="r")
    rci = np.linalg.solve(rc, np.eye(rc.shape[0]))
    sigma_c = float(np.sqrt((rci @ rci.T)[-1, -1]) / np.linalg.norm(hp))
    acc_ratio = float(np.mean(np.linalg.norm(a_pr, axis=1)) / np.mean(K2.K0 * np.linalg.norm(kern, axis=1)))
    return K._clean(dict(
        orbit=orbit, k=k, dyn=dyn, force=force, degrees=spec.get("degrees"), field_diff=spec.get("field_diff", False), mismodel_scale=eps, convolution_check_rel=conv_check,
        mean_accel_ratio_pr_over_srp=acc_ratio,
        sine_pr_vs_m3_design=sine_pr, sine_scalar_k_vs_state=sine_k,
        resid_chi2_10pct=float(eps ** 2 * np.linalg.norm(_project_out(an, hp)) ** 2),
        signal_chi2_10pct=float(eps ** 2 * hp @ hp),
        alias_10pct=dict(false_variation_chi2=float(v @ np.linalg.solve(cv, v)),
                         variation_bias=v.tolist(), variation_sd=np.sqrt(np.diag(cv)).tolist(),
                         level_bias=float(one @ df), level_bias_over_sigma=float(one @ df / math.sqrt(one @ cf @ one)),
                         state_bias_pos_m=float(np.linalg.norm(dtheta[:3]))),
        sigma_c_pr=sigma_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hf-canon", action="store_true")
    ap.add_argument("--force", choices=("PR", "GRAV"), default="PR")
    ap.add_argument("--degrees", type=int, nargs=2, default=(50, 100))
    ap.add_argument("--field-diff", action="store_true", help="GRGM1200L - GL1800F at the upper degree")
    ap.add_argument("--cells", nargs="*", default=None, help="orbit:k pairs (default: all 10)")
    a = ap.parse_args()
    cells = ([(c.split(":")[0], int(c.split(":")[1])) for c in a.cells] if a.cells
             else [(o, k) for o in K.ORBITS for k in (0, 12)])
    specs = [dict(orbit=o, k=k, dyn="SIMPLIFIED", force=a.force, degrees=tuple(a.degrees), field_diff=a.field_diff)
             for o, k in cells]
    if a.hf_canon:
        specs.append(dict(orbit="CANON", k=0, dyn=P.HF, force=a.force, degrees=tuple(a.degrees)))
    global OUT
    if a.force == "GRAV":
        OUT = K2.ARTIFACTS / ("phase17_mm_probe_gravity_fielddiff_%d.json" % a.degrees[1] if a.field_diff
                              else "phase17_mm_probe_gravity.json" if tuple(a.degrees) == (50, 100)
                              else "phase17_mm_probe_gravity_%d_%d.json" % tuple(a.degrees))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in [ex.submit(cell, s) for s in specs]:
            try:
                rows.append(fut.result())
            except Exception as exc:  # recorded
                rows.append(dict(error=repr(exc)))
    OUT.write_text(json.dumps(dict(
        label=("DIAGNOSTIC PROBE (model-mismatch pivot): linear signature of "
               + ("lunar albedo + IR radiation pressure" if a.force == "PR"
                  else "GL1800F degrees 51-100 omitted by the 50x50 estimation field")),
        model=dict(A_bond=A_BOND, T_subsolar_K=T_SS, T_night_K=T_NIGHT, approximation="sub-spacecraft point"),
        rows=rows), indent=1), encoding="utf-8")
    for r in rows:
        print({k_: r.get(k_) for k_ in ("orbit", "k", "dyn", "convolution_check_rel", "mean_accel_ratio_pr_over_srp",
                                        "sine_pr_vs_m3_design", "sine_scalar_k_vs_state", "resid_chi2_10pct",
                                        "sigma_c_pr", "error")}, r.get("alias_10pct"))


if __name__ == "__main__":
    main()
