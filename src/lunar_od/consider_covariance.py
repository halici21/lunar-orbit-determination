"""Consider covariance for a square-root least-squares posterior (Phase C2-I).

A consider parameter ``c`` (here: station clock offsets, owner decision of Phase C2 s5.3) is uncertain but not
estimated. With whitened designs ``A_x`` (state, including the a priori rows) and ``A_c`` (consider, zero on
the a priori rows) the QR factorisation of ``[A_x | A_c]`` gives

    R = [[R_xx, R_xc], [0, R_cc]],   P_x = R_xx^-1 R_xx^-T,   dx_hat/dc = R_xx^-1 R_xc,

so the sensitivity ``S = -P_x A_x^T A_c = -R_xx^-1 R_xc`` and ``P_consider = P_x + S P_cc S^T`` come out without
forming the normal matrix (Phase 17-R1COV). Columns are scaled to unit norm before the factorisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class ConsiderResult:
    p_x: np.ndarray            # (n, n) data + prior posterior covariance of the estimated state
    sensitivity: np.ndarray    # (n, q) S = -P_x A_x^T A_c (minus the response of the estimate to c)
    p_consider: np.ndarray     # (n, n) P_x + S P_cc S^T

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.diag(self.p_x))

    @property
    def sigma_consider(self) -> np.ndarray:
        return np.sqrt(np.diag(self.p_consider))


def consider_covariance(a_x_w, a_c_w, p_cc) -> ConsiderResult:
    """Consider covariance from whitened designs.

    ``a_x_w`` (m, n): rows of W^1/2 H_x, followed by the a priori rows (P0^-1/2) if the solve has a prior;
    ``a_c_w`` (m, q): W^1/2 H_c on the same rows, zero on the a priori rows; ``p_cc`` (q, q): the consider
    parameters' a priori covariance (no default: it must come from the station clock specification).
    """
    a_x = np.asarray(a_x_w, dtype=float)
    a_c = np.asarray(a_c_w, dtype=float)
    if a_c.ndim == 1:
        a_c = a_c[:, None]
    p_cc = np.atleast_2d(np.asarray(p_cc, dtype=float))
    m, n = a_x.shape
    q = a_c.shape[1]
    if a_c.shape[0] != m or p_cc.shape != (q, q):
        raise ValueError(f"shape mismatch: A_x {a_x.shape}, A_c {a_c.shape}, P_cc {p_cc.shape}.")
    if m < n:
        raise ValueError("the state design must have at least as many rows as columns.")
    sx = np.linalg.norm(a_x, axis=0)
    sc = np.linalg.norm(a_c, axis=0)
    if np.any(sx == 0.0):
        raise ValueError("a state column of the design is identically zero.")
    sc = np.where(sc == 0.0, 1.0, sc)
    r = np.linalg.qr(np.hstack([a_x / sx, a_c / sc]), mode="r")
    r_xx, r_xc = r[:n, :n], r[:n, n:n + q]
    rinv = solve_triangular(r_xx, np.eye(n), lower=False)
    p_x = (rinv / sx[:, None]) @ (rinv / sx[:, None]).T
    response = (solve_triangular(r_xx, r_xc, lower=False) / sx[:, None]) * sc[None, :]
    s = -response
    p_con = p_x + s @ p_cc @ s.T
    p_x = 0.5 * (p_x + p_x.T)
    p_con = 0.5 * (p_con + p_con.T)
    return ConsiderResult(p_x, s, p_con)
