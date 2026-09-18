"""PHASE 17-R1O - s42 covariance cross-check.

ANALYSIS SPACE ONLY.  For representative combined-observable cases,
independently verify:  QR square-root covariance ~= Schur/projector
conditional K result, i.e. 1/sqrt(I_K|x) from the orthogonal decomposition
matches sqrt(cov[K,K]) from the QR path.  This is the SAME cross-check
R1COV ran on the single range observable; here it is repeated on the new
COMBINED multi-observable systems (range+DDOR, range+landmark) to make sure
stacking new observable types into one design matrix did not silently break
the covariance path.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import relative_error  # noqa: E402
from lunar_od.constants import R_MOON_M  # noqa: E402
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_core import build_ddor_arc, build_landmark_arc  # noqa: E402
from phase17_r1o_core import K_TRUTH, scale_matrix, square_root_covariance  # noqa: E402
from phase17_r1o_core import orthogonal_decomposition, whitened  # noqa: E402

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
W15_ORBITS = 15.0
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def check(name, h_x0, h_k, w):
    scale = scale_matrix()
    a_x, b_k = whitened(h_x0, h_k, w)
    dec = orthogonal_decomposition(a_x, b_k)
    i_k_given_x = dec["i_k_given_x"]
    sigma_schur = float(1.0 / np.sqrt(i_k_given_x)) if i_k_given_x > 0 else float("inf")
    h_full = np.hstack([h_x0, h_k[:, None]])
    sr = square_root_covariance(h_full, w, None, scale)
    rel = relative_error(sr.sigma_k, sigma_schur)
    print("  %-28s QR sigma_K=%.8e  Schur sigma_K=%.8e  rel=%.3e  %s"
          % (name, sr.sigma_k, sigma_schur, rel, "PASS" if rel < 1e-6 else "FAIL"))
    return dict(case=name, qr_sigma_k=sr.sigma_k, schur_sigma_k=sigma_schur,
               relative_error=rel, passes=bool(rel < 1e-6))


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()

    hdr("s42 -- QR vs SCHUR COVARIANCE CROSS-CHECK")
    w15 = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15",
                          station_filter=ALL_STATIONS, cadence_s=90.0)
    rows = [check("range_only_W15", w15.h_x0, w15.h_k, w15.w)]

    ddor = build_ddor_arc(w15.nom48, w15.t_grid, et0,
                          ("Goldstone DSN", "Canberra DSN"),
                          sigma_angle_rad=5e-9, label="ddor_5nrad")
    h_x0 = np.vstack([w15.h_x0, ddor.h_x0])
    h_k = np.concatenate([w15.h_k, ddor.h_k])
    w = np.concatenate([w15.w, ddor.w])
    rows.append(check("range_plus_ddor_5nrad", h_x0, h_k, w))

    lm = build_landmark_arc(w15.nom48, w15.t_grid, et0, sigma_angle_rad=30e-6,
                            r_moon_m=R_MOON_M, label="landmark_30urad")
    h_x0 = np.vstack([w15.h_x0, lm.h_x0])
    h_k = np.concatenate([w15.h_k, lm.h_k])
    w = np.concatenate([w15.w, lm.w])
    rows.append(check("range_plus_landmark_30urad", h_x0, h_k, w))

    with (ARTIFACTS / "r1o_covariance_crosscheck.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)

    gate = all(r["passes"] for r in rows)
    print("\n  R1O_COVARIANCE_CROSSCHECK_GATE = %s" % ("PASS" if gate else "FAIL"))
    (ARTIFACTS / "r1o_crosscheck_summary.json").write_text(
        json.dumps(dict(gate="PASS" if gate else "FAIL", cases=rows),
                  indent=2, default=float))
    print("  wrote r1o_covariance_crosscheck.csv, r1o_crosscheck_summary.json")


if __name__ == "__main__":
    main()
