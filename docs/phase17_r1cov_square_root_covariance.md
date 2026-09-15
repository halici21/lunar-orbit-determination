# PHASE 17-R1COV — SQUARE-ROOT COVARIANCE QUALIFICATION AND K_SRP FORMAL-UNCERTAINTY REPAIR

## 1. Executive Summary

R1M established that the production-reported `sigma_K` was controlled by the eigenvalue safety
floor rather than by the data or the prior, and set
`ESTIMATOR_SIGMA_K_TRUST_STATUS = NOT_RELIABLE_IN_NEAR_SINGULAR_REGIME`. R1COV asked whether the
formal covariance can instead be recovered from a square-root representation of the weighted design
problem, and whether that can be integrated without disturbing anything already qualified.

Both answers are yes.

**The defect reproduced exactly, on the shipped code.** Against an exact oracle, the production
path was overconfident by **260.6×** (G0), **11.8×** (G1) and **10.2×** (G3). Exactly one
eigenmode was clipped in each case and it was essentially pure K. The condition-squaring relation
`cond(H^T W H) / cond(H_w)^2` measured `1.0000000288`, `0.9999999981` and `1.0000000011` — the
normal matrix's ill-conditioning is manufactured, not inherited.

**The oracle is exact rather than merely high-precision.** `mpmath` is not installed, and adding a
dependency to this frozen repository was not authorised. The standard library offers something
stronger: every IEEE-754 double is a dyadic rational, so `fractions.Fraction` represents the entire
normal matrix exactly and an exact Gauss-Jordan inverse carries **zero** arithmetic error. The §18
convergence study still ran, in `Decimal` at 50/80/120/200 digits, and every level reproduced the
exact answer to the last bit of its float64 rendering. This matters because the question — is the
weak K direction destroyed by arithmetic or absent from the data? — is settled by an exact oracle
and only bounded by a high-precision one.

**The square-root path is qualified.** Building `A = [W^(1/2) H S ; L^T]`, factoring `A = QR` and
recovering `P = R^-1 R^-T` by two triangular solves reproduces the exact covariance to `≤1.6e-13`
on `sigma_K` and `≤6.8e-12` across the full matrix. Scaling invariance — the hard gate of §27 and
§63 — passes with a spread of **exactly `0.000e+00`** across K scales spanning 4×, where the old
path's spread was `3.000e+00`. The prior response gate closes the R1/R1G equal-sigma pathology
directly: broad and moderate priors differed by `3.4e-15` (i.e. were identical) under the old path
and by a factor of **66** under the repaired one.

**A second contaminated output was found.** `_sqrt_information_from_information`, which produces
the SRIF's reported `posterior_sqrt_information`, applies the *same* floor and then factors the
already-floored matrix. It is a square root **of** the defect, wrong by 90–99.6%. It is repaired on
the K path only.

**Production integration was minimal and is bitwise-safe.** A call-site audit found 28 uses of the
floored helpers; exactly **2** were in scope and modified. The point estimate is unchanged
**bitwise** — K estimate, final cost, iteration count and the full state vector all compare
`max|diff| = 0.000e+00` against the pre-repair code executed against byte-identical physics. The
default six-state paths, the `scenarios.py` duplicate helper, and SR-UKF are untouched; SR-UKF was
audited and proven structurally immune, since it carries `sqrt_p` directly and never forms a normal
matrix.

**What this does not do is rescue K.** The honest numbers are worse than the ones they replace.
G0's `sigma_K` is `0.904` against a truth of `0.01` — K is unconstrained there. R1M's best
configuration now reports `4.516e-03`, i.e. **45.2% fractional uncertainty, 2.21 sigma** from zero,
which exactly reproduces R1M's information-domain value and so confirms R1M's conclusion rather
than overturning it. The repair makes the weakness visible; it does not remove it.

```
ESTIMATOR_SIGMA_K_TRUST_STATUS_AFTER_R1COV = QUALIFIED
    for the two explicitly qualified K solve-for paths (BLS, SRIF).
    The default six-state paths retain the floored helper and are NOT
    covered by this qualification.
```

## 2. What R1M Discovered

R1M traced the covariance pipeline and found the reported `sigma_K` was neither the data answer nor
the prior answer. `sqrt(K_scale^2 / floor)` reproduced it to `1.5e-10` using no data quantity but
the largest eigenvalue; it moved linearly with an arbitrary bookkeeping constant while every
physical quantity stayed bitwise invariant; and exactly one eigenmode — essentially pure K — was
clipped, by 67,917× on G0.

R1M also measured that `cond(H^T W H) = cond(H_w)^2` to a ratio of 1.0000, so the rank deficiency
that had been read as "K is unobservable" was partly a property of the representation. The design
matrix sat at `1e8`–`2.6e9` against a double-precision limit of `4.5e15`.

## 3. Why R1COV Is Required Before More K Science

Any further observability experiment reports a `sigma_K`. Until the covariance path is qualified,
that number is an artifact, and an experiment whose headline quantity is an artifact cannot produce
a scientific conclusion — it can only produce a confident-looking one. R1COV therefore precedes
R1O by necessity rather than convenience.

The distinction that makes this a separate phase rather than a footnote is that a parameter
*estimate* and a parameter *uncertainty* come from different machinery. R1/R1-R qualified the
estimate. Nothing had qualified the uncertainty.

## 4. Academic Context

The governing result is elementary and long established: for the normal equations,

```
kappa(A^T A) = kappa(A)^2
```

so forming `A^T A` costs half the available significant digits. Björck's treatment of numerical
least squares is the standard reference; the recommended response to an ill-conditioned
least-squares problem is orthogonalization rather than normal equations.

In spacecraft navigation the same reasoning produced the square-root information filter. Bierman's
*Sequential Least Squares Using Orthogonal Transformations* (JPL TM-33-735) and *Factorization
Methods for Discrete Sequential Estimation* introduced the R-factor formulation precisely so that
orbit determination would not lose conditioning to squaring, and SRIF has been operational in JPL
navigation ever since. Wang et al.'s applications of SRIF and smoothing to spacecraft orbit
determination are the direct lineage of the estimator this repository already ships.

Nothing about the square-root method is novel here, and this report claims no novelty for it. What
R1COV contributes is a measured instance: `cond(I)/cond(A)^2 = 1.0000` to eight digits on real
lunar two-way range design matrices, with the resulting covariance error quantified against an
exact oracle at 260×.

## 5. Industry / Navigation Context

The operational hazard is the specific shape of this failure. The old covariance was finite,
positive definite, symmetric, and numerically plausible. Nothing about `sigma_K = 0.00347` looks
wrong. It was wrong by 260×, in the optimistic direction, and it would have propagated silently
into anything that consumed it.

The second hazard is the obvious remedy. Replacing an eigenvalue floor with a default
`np.linalg.pinv` inverts the failure rather than fixing it: the default `rcond` discards the weak
singular direction, K's variance contribution vanishes, and the method reports essentially zero
uncertainty on an essentially unconstrained parameter. R1M measured `5.5e-10`; §6 forbids it, and
§21 of this report measures it again on controlled problems.

## 6. Entering Repository State

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = f0ee0fac0f07db8ad3da60ef7434749eaa5438a4
START_TREE   = c646b86d9281d4053155a686c80fa1e7bfbbf737
```

`R1M_CONTENT_COMMIT_REACHABLE = YES` (`f2af3045e44860b0892fe84b788ab7b75c7c63a2`, verified by
`git merge-base --is-ancestor`). Canonical `main` and `origin/main` both verified unchanged at
`fea476f81dad07b3914e53eab709e10fa6e9d10b`. Working tree clean at entry.

## 7. R1M Defect Reproduction

**Purpose.** Confirm the defect on the shipped code before proposing a repair for it.

**Numerical hypothesis.** The reported `sigma_K` is produced by the eigenvalue floor, and the
floor engages because forming the normal matrix squares an already-large condition number.

**Setup.** G0 (1.3 orbits, 82 obs), G1 (2.0, 144), G3 (5.0, 307) on the frozen campaign arc;
`K_truth = 0.01`, cadence 60 s, `rtol=1e-12`, `atol=1e-13`, noise-free two-way converged-event
range, scaling `diag(1e6,1e6,1e6,1e3,1e3,1e3,1e-2)`. The production helper
`_safe_covariance_from_information` is called directly so the reproduction cannot drift from
shipped behaviour.

**Method.** Compare four routes: production floored inverse, exact rational inverse, data-only
scalar Schur complement, and the condition numbers of design versus normal matrix.

**What changed.** Nothing — physics, measurement model, point estimator and numerics are all the
frozen production code. Only the analysis harness is new.

**Results.**

| | G0 | G1 | G3 |
|---|---:|---:|---:|
| `cond(design)` | 2.606087e+09 | 1.179137e+08 | 1.017109e+08 |
| design resolvable (`< 4.5e15`) | yes | yes | yes |
| `cond(information)` | 6.791689e+18 | 1.390364e+16 | 1.034510e+16 |
| information resolvable | **no** | **no** | **no** |
| `cond(I) / cond(A)^2` | 1.0000000288 | 0.9999999981 | 1.0000000011 |
| smallest information eigenvalue | 1.223619e-04 | 2.658520e-01 | 3.520275e+00 |
| floor (`max_eig * 1e-14`) | 8.310441e+00 | 3.696310e+01 | 3.641760e+02 |
| modes clipped | 1 of 7 | 1 of 7 | 1 of 7 |
| floor inflation | **67,917×** | **139.0×** | **103.5×** |
| production `sigma_K` | 3.4688695056e-03 | 1.6448102377e-03 | 5.2401571115e-04 |
| exact `sigma_K` | 9.0401755679e-01 | 1.9394564345e-02 | 5.3298100020e-03 |
| data-only Schur `sigma_K` | 9.0401752984e-01 | 1.9394564407e-02 | 5.3298099961e-03 |
| **overconfidence factor** | **260.6×** | **11.8×** | **10.2×** |

The exact inverse and the independent scalar Schur complement agree to `3e-8` or better. The
production path disagrees with both, always optimistically.

**Interpretation.** Confirmed on shipped code. The design matrix is comfortably resolvable; the
normal matrix is not; the gap is exactly the square.

**System impact.** None — diagnostic only.

**Academic interpretation.** A clean measured instance of `kappa(A^T A) = kappa(A)^2`.

**Industry interpretation.** A plausible-looking formal sigma, wrong by 260×.

**Limitations.** Three arcs of one geometry; the mechanism is general, the magnitudes are not.

**Verdict.** `PASS`. `R1M_COVARIANCE_DEFECT_REPRODUCED = YES`,
`NORMAL_MATRIX_CONDITION_SQUARING_GATE = PASS`.

## 8. Current Covariance Call Graph

Every use of the two floored helpers was located by AST-attributed source scan before any edit.

| classification | call sites | in scope for R1COV |
|---|---:|:--:|
| `DEFAULT_6STATE_ESTIMATOR` | 8 | no |
| `K_AUGMENTED_BLS` | 1 | **yes** |
| `K_AUGMENTED_SRIF` | 1 | **yes** |
| `OTHER_ESTIMATOR_PATH` (`scenarios.py` duplicate helper) | 1 | no |
| `TEST_UTILITY` | 3 | no |
| `ANALYSIS_ONLY_CALLER` | 14 | no |
| **total** | **28** | **2** |

`scenarios.py` carries its own separate *definition* of `_safe_covariance_from_information`. It is
not part of the K solve-for chain and was not touched (§39). The full inventory with file, line,
enclosing function and classification is `artifacts/r1cov_callsite_inventory.csv`.

## 9. Existing Floor Semantics

```python
def _safe_covariance_from_information(information):
    vals, vecs = np.linalg.eigh(_symmetrize(information))
    floor = max(float(np.max(np.abs(vals))) * 1e-14, np.finfo(float).eps)
    vals = np.clip(vals, floor, None)
    return _symmetrize((vecs / vals) @ vecs.T)
```

Two properties matter. The floor is **relative to the largest eigenvalue**, so on a matrix with a
`1e18` dynamic range it is enormous in absolute terms. And it is an **absolute threshold applied to
a scale-dependent matrix**, so the clipped value — and hence the reported variance — inherits the
arbitrary scaling. That is the mechanism behind R1M's linear-in-`K_scale` observation.

As a safety net for a well-conditioned six-state problem it is unobjectionable and it is retained
there. As the determinant of a weak parameter's variance it is not a safety net at all; it is the
answer.

## 10. High-Precision Oracle

**Purpose.** Obtain a reference that is not itself float64.

**Method.** `fractions.Fraction`. Every finite double is a dyadic rational `p/2^k`; sums and
products of dyadic rationals are dyadic rationals; so `H^T W H + P0^-1` is exactly representable
and an exact Gauss-Jordan inverse has zero arithmetic error.

The oracle is exact for the linear algebra only. It inherits whatever error is already in `H` and
`W` from propagation and measurement partials. It is the exact covariance *of the design matrix we
were given* — which is precisely the reference needed to qualify a covariance **algorithm**, and
not a claim about the physical truth of `H`.

**Control (§19).** On a deliberately benign problem (`cond(A) = 100`, `cond(A^T A) = 1.0e4`) the
exact oracle must agree with an ordinary float64 inverse, or it is simply wrong.

| metric | value |
|---|---:|
| entrywise relative error | 1.461e-12 |
| correlation-normalised error | **7.009e-14** |
| float64 accuracy limit `kappa(A^T A) * eps` | 2.220e-12 |
| observed / theoretical limit | **0.0316** |

The first evaluation of this gate FAILED, against a threshold of `1e-12` that I had chosen rather
than derived. Two things were wrong with it and both are recorded rather than quietly corrected.
The entrywise metric divides by `|cov[i,j]|`, so a covariance entry that happens to lie near zero
reports a large relative error for a negligible absolute one — the smallest entry here sits at
`3.0e-2` of its own natural scale. And the threshold demanded that float64 agree more closely than
float64 can compute: inverting a normal matrix of condition `1e4` cannot beat `kappa * eps ≈
2.2e-12`. The gate now uses the correlation-normalised metric (the same one used for every other
comparison in this phase) against a threshold **derived** from that limit. Even the discarded
entrywise figure was inside the theory bound.

**Verdict.** `PASS`.

## 11. Oracle Precision Convergence

`Decimal` at four precisions, against the exact rational limit:

| case | exact `sigma_K` | 50 digits | 80 digits | 120 digits | 200 digits | 80→120 change |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 9.0401755678754248e-01 | = | = | = | = | 0.000e+00 |
| G1 | 1.9394564345072467e-02 | = | = | = | = | 0.000e+00 |
| G3 | 5.3298100019765473e-03 | = | = | = | = | 0.000e+00 |

Every precision level reproduces the exact value to the last bit of its float64 rendering. Five
independent arithmetic systems — exact rationals and four Decimal precisions — agree, and the
data-only Schur complement agrees to `3e-8` by a sixth route.

`HIGH_PRECISION_ORACLE_CONVERGENCE_GATE = PASS`.

## 12. QR Square-Root Formulation

The weighted augmented system is built directly:

```
A = [ W^(1/2) H S ]        S = diag(1e6,1e6,1e6,1e3,1e3,1e3,K_scale)
    [    L^T      ]        L L^T = S^T P0^-1 S
```

The prior enters as appended **square-root rows**, not as `P0^-1` inside a normal matrix (§22).
`L^T` is obtained by symmetric eigendecomposition rather than Cholesky, because the prior
information matrix here is routinely only semi-definite — the data-only case is all zeros and a
partial prior constrains some parameters and not others, and Cholesky raises on both.

Then `A = QR`, and `P_s = R^-1 R^-T` is recovered by **two triangular solves** rather than an
explicit inverse (§24):

```
R^T Y = I      (lower-triangular solve)
R   P = Y      (upper-triangular solve)
```

`R^T R` is never formed, and neither is `H^T W H` on this path. Unpivoted QR proved sufficient
after the established scaling; because no pivoting is used, parameter identity is preserved
trivially and the K index needs no unpermuting, so no covariance unpermutation can go wrong (§25).

§25 also asks for a rank-revealing control. Rather than column-pivoted QR, the control used here is
the **singular value decomposition of R**, which is the reference rank-revealing factorization
rather than an approximation to one — pivoted QR reveals rank only up to a bound, whereas the SVD
gives the singular values themselves. Since `R` is 7×7 the cost is irrelevant. The rank criterion in
section 20 is applied to those singular values.

## 13. QR vs High-Precision Comparison

Relative error of the square-root path against the exact oracle, across all three prior cases:

| case | data-only | broad prior (σ=1.0) | moderate prior (σ=0.01) |
|---|---:|---:|---:|
| G0 | 1.55e-13 | 8.53e-14 | 0.00e+00 |
| G1 | 7.03e-14 | 7.05e-14 | 1.48e-14 |
| G3 | 1.89e-14 | 1.87e-14 | 1.49e-14 |

Full-covariance agreement (max correlation-normalised error): `6.74e-12` (G0), `1.60e-13` (G1),
`6.81e-14` (G3), including the K-state cross terms and the state sigmas.

`QR_COVARIANCE_ORACLE_GATE = PASS`.

## 14. Scaling Invariance

The hard gate (§27, §63). `K_scale` is a bookkeeping constant; a physical uncertainty may not
depend on it.

| case | square-root `sigma_K` spread | floored `sigma_K` spread |
|---|---:|---:|
| G0 | **0.000e+00** | 3.000e+00 |
| G1 | **0.000e+00** | 3.000e+00 |
| G3 | **0.000e+00** | 3.000e+00 |

The repaired path is invariant to the last bit across K scales spanning 4×. The old path's spread of
exactly `3.000` is the scale ratio itself (`2e-2 / 5e-3 = 4`, giving `4 - 1 = 3`), which is the
linear dependence R1M identified, reproduced here independently.

`QR_COVARIANCE_SCALING_INVARIANCE_GATE = PASS`.

## 15. Prior Response

| case | prior | old (floored) | repaired | exact oracle |
|---|---|---:|---:|---:|
| G0 | data-only | 3.468870e-03 | 9.040176e-01 | 9.040176e-01 |
| G0 | broad σ=1.0 | 3.468870e-03 | 6.706096e-01 | 6.706096e-01 |
| G0 | moderate σ=0.01 | 3.468870e-03 | 9.999388e-03 | 9.999388e-03 |
| G1 | data-only | 1.644810e-03 | 1.939456e-02 | 1.939456e-02 |
| G1 | broad σ=1.0 | 1.644810e-03 | 1.939092e-02 | 1.939092e-02 |
| G1 | moderate σ=0.01 | 1.644810e-03 | 8.888092e-03 | 8.888092e-03 |
| G3 | data-only | 5.240157e-04 | 5.329810e-03 | 5.329810e-03 |
| G3 | broad σ=1.0 | 5.240157e-04 | 5.329734e-03 | 5.329734e-03 |
| G3 | moderate σ=0.01 | 5.240157e-04 | 4.703461e-03 | 4.703461e-03 |

Broad-versus-moderate separation, old → repaired: G0 `3.38e-15 → 66.1`, G1 `9.23e-16 → 1.18`,
G3 `0.00e+00 → 0.133`.

The old column is constant down each case: the prior was not reaching the answer at all. This is the
direct closure of the R1/R1G equal-sigma pathology, and it also confirms R1M's retrospective
explanation of it.

The physics now reads correctly. On G0, where the data are nearly uninformative, a σ=0.01 prior
yields a posterior of `9.999e-03` — the posterior is the prior, because the data added almost
nothing. On G3, where the data are stronger, the same prior moves the posterior only from
`5.330e-03` to `4.703e-03`.

`QR_PRIOR_RESPONSE_GATE = PASS`.

## 16. Existing SRIF R-Factor Audit

§29 warns against assuming the shipped SRIF factor is usable as a posterior covariance square root.
The warning is well placed. The shipped helper is:

```python
def _sqrt_information_from_information(information):
    vals, vecs = np.linalg.eigh(_symmetrize(information))
    floor = max(float(np.max(np.abs(vals))) * 1e-14, np.finfo(float).eps)
    if np.min(vals) <= floor:
        vals = np.clip(vals, floor, None); information = rebuild(...)
    return np.linalg.cholesky(information).T
```

It takes the already-squared normal matrix, applies the **same** floor, and only then factors. It
is a square root **of the defect**:

| case | exact `sigma_K` | from shipped sqrt-information | relative error |
|---|---:|---:|---:|
| G0 | 9.040176e-01 | 3.468870e-03 | **9.96e-01** |
| G1 | 1.939456e-02 | 1.644810e-03 | **9.15e-01** |
| G3 | 5.329810e-03 | 5.240157e-04 | **9.02e-01** |

`SHIPPED_SQRT_INFORMATION_CONTAMINATED = YES`. This is a second instance of the same defect, in a
second reported field, and it was not visible in R1M.

## 17. SRIF Covariance Reconstruction

A genuine R factor was accumulated **sequentially**, one measurement row at a time, through
successive orthogonal updates — the classical square-root information measurement update, which
never forms the normal matrix at any point. Sequential accumulation is a materially different
sequence of floating-point operations from the batch QR the BLS path uses, so agreement between
them is evidence rather than tautology.

**Analytic control (§30).** On a problem whose posterior is known in closed form (diagonal
information, analytic inverse), the exact oracle, batch QR and sequential SRIF all reproduce the
analytic `sigma_K`, and the full covariance matches to `<1e-12` normalised.

| case | exact | batch QR | sequential SRIF | QR rel | SRIF rel |
|---|---:|---:|---:|---:|---:|
| G0 | 9.040176e-01 | 9.040176e-01 | 9.040176e-01 | 1.55e-13 | 5.21e-14 |
| G1 | 1.939456e-02 | 1.939456e-02 | 1.939456e-02 | 7.03e-14 | 1.39e-13 |
| G3 | 5.329810e-03 | 5.329810e-03 | 5.329810e-03 | 1.89e-14 | 1.87e-13 |

`SRIF_R_FACTOR_COVARIANCE_GATE = PASS`.

## 18. BLS–SRIF Covariance Comparison

Full-covariance cross-agreement (max correlation-normalised error), including K-state cross terms:

| case | batch QR vs exact | sequential SRIF vs exact | QR vs SRIF |
|---|---:|---:|---:|
| G0 | 6.74e-12 | 2.56e-12 | 9.28e-12 |
| G1 | 1.60e-13 | 1.58e-12 | 1.68e-12 |
| G3 | 6.81e-14 | 3.95e-13 | 3.39e-13 |

`BLS_SRIF_COVARIANCE_CONSISTENCY_GATE = PASS`.

## 19. Weak-Full-Rank Control

Controlled 7-parameter designs whose seventh direction has a prescribed singular value:

| weak singular value | rank margin | exact `sigma_K` | square-root | floored | floored error |
|---|---:|---:|---:|---:|---:|
| 1e-3 | 6.43e+11 | 1.509974e+02 | 1.509974e+02 | 1.509974e+02 | 1.26e-10 |
| 1e-5 | 6.43e+09 | 1.509915e+04 | 1.509915e+04 | 1.509913e+04 | 1.39e-06 |
| 1e-7 | 6.43e+07 | 1.509915e+06 | 1.509915e+06 | 1.509915e+06 | 9.25e-11 |
| 1e-9 | 6.43e+05 | 1.509915e+08 | 1.509915e+08 | 1.509915e+06 | **9.90e-01** |

The square-root path tracks the exact answer throughout; the floored path collapses at the weakest
case, understating by 100×. The understatement grows as the direction weakens — the failure is
worst exactly where a correct uncertainty matters most.

`WEAK_DIRECTION_RETENTION_GATE = PASS`.

## 20. Exact-Rank-Deficient Control

Two constructions, because the first attempt conflated two different things and the distinction
turned out to be the subject of the phase.

**E2a — bitwise duplicate column.** Exactly dependent as a rational. The exact oracle raises
`exactly singular`; the square-root path raises `RankDeficientCovarianceError` (rank 6 of 7, ratio
`8.07e-17` against threshold `1.55e-15`); the floored path returns a finite `7.52e+06`; default
`pinv` returns `1.26`. Both of the latter fabricate a number for a direction carrying zero
information.

**E2b — dependent column computed in float64.** My first version of this control built the
dependent column as `a[:, :6] @ coeffs`, and the exact oracle correctly found it *non*-singular:
rounding leaves the result outside the exact span. I had expected a refusal and got a finite
`7.87e+15`, so I tested whether that number means anything, by evaluating the same column three
algebraically identical ways:

| evaluation | exact `sigma_K` |
|---|---:|
| `A @ c` | 7.871354e+15 |
| forward sum | 6.507593e+15 |
| reverse sum | 4.940497e+15 |

A **1.59× spread** from summation order alone. The finite value is set by rounding, not by data, so
refusing to report it is correct and my expectation was wrong, not the code. The square-root path
refuses both E2a and E2b.

`TRUE_RANK_DEFICIENCY_SEMANTICS_GATE = PASS`.

**Threshold policy (§36).** Rank is decided on the **singular values of R** by
`sigma_i > sigma_max * n * eps` — applied to singular values rather than `|diag(R)|`, because a
triangular factor's diagonal can be arbitrarily unrepresentative of them. It is a pure ratio, so it
is invariant under uniform rescaling (verified across six orders of magnitude: rank and ratio
identical), which is exactly the scale-awareness the absolute floor lacked. On the real lunar arcs
the ratio is `~3.8e-10` against a threshold of `~1.6e-15` — five orders of margin, so the weak K
direction is retained on its own merits and not by a generous cutoff.

## 21. SVD / Pseudoinverse Characterization

| weak singular value | exact | square-root | floored | default `pinv` |
|---|---:|---:|---:|---:|
| 1e-3 | 1.509974e+02 | 1.509974e+02 | 1.509974e+02 | 1.509974e+02 |
| 1e-5 | 1.509915e+04 | 1.509915e+04 | 1.509913e+04 | 1.509913e+04 |
| 1e-7 | 1.509915e+06 | 1.509915e+06 | 1.509915e+06 | 1.517672e+06 |
| 1e-9 | 1.509915e+08 | 1.509915e+08 | 1.509915e+06 | **1.324928e+00** |

At the weakest case the default pseudoinverse is wrong by **eight orders of magnitude**, in the
dangerous direction: it reports near-zero uncertainty on a nearly unconstrained parameter, because
default `rcond` discarded the K singular direction entirely. Confirmed as unusable here (§6, §37).
SVD is retained as a *diagnostic* for singular-value inspection, never as a covariance oracle with
an unexamined cutoff.

## 22. Production Impact Audit

Covered in section 8. Two call sites in scope: the `solve_for_k_srp=True` branches of
`estimate_two_way_range_bls_lm` and `estimate_two_way_range_srif`. Phase 17-R had already isolated
both behind a `solve_for_k_srp` conditional, so the six-state branches are reached by a different
code path entirely and were not edited.

## 23. Controlled Production Integration

Changes to `lunar_od/estimators.py`, and nothing else:

1. **`RankDeficientCovarianceError`** — new exception carrying `rank`, `n`, `singular_ratio`,
   `threshold`.
2. **`_square_root_covariance_from_design`** — new helper returning
   `(covariance_physical, sqrt_information_physical)`. The R factor is mapped back to physical
   coordinates by column division (the scale is diagonal, so upper-triangularity is preserved),
   giving `R_phys^T R_phys = posterior_information`.
3. **`_two_way_range_posterior_design_with_k`** — split out of
   `_two_way_range_posterior_information_with_k` so the covariance path can reach the design rows
   without a second trajectory propagation. The information matrix's arithmetic is unchanged by the
   split, so `posterior_information` remains bitwise identical to Phase 17-R.
4. **Two K branches** now derive covariance from the design; `posterior_information` is still
   computed and still reported, it is simply no longer *inverted*.
5. **SRIF `posterior_sqrt_information`** takes the genuine R factor on the K path only; the
   six-state path keeps `_sqrt_information_from_information` untouched. Shipping a qualified
   covariance beside a contaminated sqrt-information in the same stats object would have been a trap.
6. `from scipy.linalg import solve_triangular` — scipy is already a hard dependency of `lunar_od`
   (`dynamics`, `filters`, `ephemeris`), so this adds no new requirement.

`_safe_covariance_from_information` itself is **unmodified**.

## 24. G0 Requalification

| | value |
|---|---:|
| old reported `sigma_K` | 3.468870e-03 |
| qualified `sigma_K` | **9.040176e-01** |
| exact oracle | 9.040176e-01 (rel 1.5e-13) |
| `sigma_K / K_truth` | **90.4** |

K is not merely weakly determined on this arc; it is unconstrained. The old figure was 260× too
confident.

## 25. G1 Requalification

| | value |
|---|---:|
| old reported `sigma_K` | 1.644810e-03 |
| qualified `sigma_K` | **1.939456e-02** |
| exact oracle | 1.939456e-02 (rel 7.0e-14) |
| `sigma_K / K_truth` | **1.94** |

G1 was R1G's "strongest stable case". Its true formal uncertainty is 194% of the truth value.

A live end-to-end BLS solve confirms the integration: the same run reports `sigma_K` of
`1.6447674e-03` before the repair and `1.9392789e-02` after, a factor of 11.79, matching the
standalone requalification.

## 26. Prior Diagnostic Requalification

Covered in section 15 and reproduced through the production helper in COV-G.
`PRIOR_DIAGNOSTIC_POST_REPAIR_GATE = PASS`.

## 27. Best Existing Linked-Span Characterization

R1M's best configuration, recomputed with the qualified covariance rather than the
information-domain proxy `1/sqrt(I(K|x))`:

| | value |
|---|---:|
| configuration | M3 Model B: five 0.5-orbit windows across a 5.50-orbit span |
| observations | 295 |
| old reported `sigma_K` | 4.503114e-04 |
| **qualified `sigma_K`** | **4.516318e-03** (oracle rel 5.2e-14) |
| `K_truth` | 1.000000e-02 |
| fractional uncertainty | **45.2%** |
| separation from zero | **2.21 sigma** |
| old understatement | 10.03× |

This reproduces R1M's information-domain value of `4.516e-03` exactly. R1M's conclusion therefore
**stands on the qualified covariance**, not merely on a proxy.

```
BEST_EXISTING_RADIOMETRIC_K_UNCERTAINTY_CLASS =
  NUMERICALLY_OBSERVABLE_BUT_NOT_PRACTICALLY_IDENTIFIABLE
```

## 28. Point-Estimate Invariance

The pre-repair `estimators.py` was loaded from git as a sibling module bound to the **real**
`lunar_od` package, so both versions ran against byte-identical physics, measurement and dynamics
code. (Copying the whole package does not work: the copy defines its own `TwoWayRangeConfig` and
`PassGeometry`, and objects built by the real package then fail the copy's isinstance checks.)

| quantity | pre-repair | repaired | equal |
|---|---:|---:|:--:|
| K estimate | 1.03841078202146627e-03 | 1.03841078202146627e-03 | **bitwise** |
| final cost | 5.41374304355888744e+05 | 5.41374304355888744e+05 | **bitwise** |
| iterations | 12 | 12 | yes |
| state vector | — | — | **bitwise** (`max|diff| = 0.000e+00`) |
| `sigma_K` | 1.6447674e-03 | 1.9392789e-02 | changed, 11.79× |

`POINT_ESTIMATE_INVARIANCE_GATE = PASS`. The estimate is untouched; only the uncertainty moved.
This is structural as well as measured — covariance is computed after the solve loop and never
feeds back into it.

## 29. Default Six-State Protection

The six-state branches call `_safe_covariance_from_information` exactly as before, on an
unmodified helper, reached through a different conditional branch. A dedicated test asserts the
helper still reproduces its historical floored output bit for bit.

`BLS_DEFAULT_PARITY`, `SRIF_DEFAULT_PARITY`, `SRUKF_DEFAULT_PARITY` all `PASS`;
`DEFAULT_6STATE_BEHAVIOR_CHANGED = NO`.

## 30. Historical Result Correction Matrix

Historical artifacts are left intact as historical evidence (§46); this table is additive.

| Historical claim | Old basis | R1COV status | Correct interpretation |
|---|---|---|---|
| R1/R1G G0 `sigma_K = 3.47e-03` | floored normal-matrix inverse | SUPERSEDED | `9.040e-01`, 261× larger; K unconstrained (9040% of truth) |
| R1G G1 `sigma_K = 1.64e-03` | floored normal-matrix inverse | SUPERSEDED | `1.939e-02`, 11.8× larger; 194% of truth |
| R1G G3 `sigma_K = 5.24e-04` | floored normal-matrix inverse | SUPERSEDED | `5.330e-03`, 10.2× larger; 53% of truth |
| "rank 6/7" | numerical rank of `H^T W H` in float64 | REINTERPRETED | artifact of forming the normal matrix; the design matrix is full rank with ~5 orders of margin. K is WEAK_BUT_RESOLVABLE, not absent from the data |
| broad and moderate priors give identical `sigma_K` | both reported the floor | RESOLVED | neither prior reached the answer; qualified posteriors differ by 66× on G0 |
| prior classed DIAGNOSTIC_UNCERTAIN / PRIOR_INFLUENCED / PRIOR_DOMINATED | inference from an uninterpretable sigma | SUPERSEDED | prior influence can now be read directly, against an exact oracle |
| R1M: `sigma_K` NOT_RELIABLE_IN_NEAR_SINGULAR_REGIME | R1M covariance audit | CONFIRMED_AND_REPAIRED | confirmed independently; repaired on the K path; default six-state path unchanged and still floored |
| R1M best case `sigma_K ≈ 4.52e-03` from `1/sqrt(I(K|x))` | information-domain proxy | **CONFIRMED** | qualified covariance reproduces it exactly; R1M's conclusion stands |

Machine-readable at `artifacts/r1cov_historical_correction_table.csv`.

## 31. What Changed in the System

```
PRODUCTION_CODE_CHANGED         = YES  (lunar_od/estimators.py only)
POINT_ESTIMATE_BEHAVIOR_CHANGED = NO   (bitwise verified)
PHYSICAL_MODEL_CHANGED          = NO
MEASUREMENT_MODEL_CHANGED       = NO
FORMAL_COVARIANCE_METHOD_CHANGED= YES  (K solve-for paths only)
DEFAULT_6STATE_BEHAVIOR_CHANGED = NO
```

Precisely which outputs now differ, and why:

- `EstimatorStats.posterior_covariance` when `solve_for_k_srp=True`, for BLS and SRIF: now from an
  orthogonal factorization of the design matrix instead of a floored eigen-inverse of the normal
  matrix. Values are 10–260× larger in the tested regime because the old ones were floor-limited.
- `EstimatorStats.posterior_sqrt_information` when `solve_for_k_srp=True`, for SRIF: now the genuine
  R factor instead of a Cholesky of the floored normal matrix.
- A rank-deficient augmented problem now raises `RankDeficientCovarianceError` where it previously
  returned a floor-derived finite covariance.

Everything else — every default six-state output, every point estimate, every residual, every
`posterior_information` matrix — is bitwise unchanged.

## 32. What Did Not Change

`_safe_covariance_from_information` and `_sqrt_information_from_information` are unmodified.
The `scenarios.py` duplicate helper is unmodified. SR-UKF is unmodified. No physics, measurement
model, event solver, or SRP code was touched. No rank tolerance was tuned to change a reported
rank. No historical report or artifact was rewritten. No new observable was added.

## 33. Scientific Interpretation

The central distinction this phase enforces is between **numerical observability** and **practical
identifiability** (§47).

K_SRP is numerically observable in the tested geometry: the whitened design matrix resolves the K
direction with five orders of margin above the resolvability threshold, and the conditional
information is strictly positive and grows with arc length. The earlier reading of "rank 6/7" as
"K is absent from the data" was a statement about the normal-matrix representation, not about the
data.

K_SRP is not practically identifiable in that geometry. The best configuration available from
existing radiometrics leaves 45.2% fractional uncertainty, about 2.21 sigma from zero. G0 leaves
9040%. Both numbers are now trustworthy, and both are worse than what was previously reported.

That is the uncomfortable shape of this result and it should be stated plainly: the repair moved
every headline number in the unhelpful direction. R1's and R1G's optimistic sigmas were the reason
K observability looked closer to viable than it is. Honest covariance does not rescue a weak
parameter; it reveals how weak it was.

## 34. Academic Counterpart

*Why is forming normal equations dangerous here?* Because `kappa(A^T A) = kappa(A)^2` and the
design matrices sit at `1e8`–`2.6e9`. Squaring lands them at `1e16`–`6.8e18`, past the `4.5e15`
double-precision limit, so a direction that is resolvable becomes unresolvable purely through the
choice of representation. This was measured, not assumed: ratio `1.0000` to eight digits.

*Why are QR / SRIF methods appropriate?* Orthogonal transformations preserve the 2-norm condition
number, so the factorization inherits `kappa(A)` rather than its square. This is the reasoning that
produced SRIF in JPL navigation, and it applies here unchanged.

*How does the measured result relate to established theory?* It is a confirming instance, with the
consequence quantified end-to-end: a 260× covariance error traceable to the squaring.

*Is the floor-artifact finding scientific novelty or software qualification?* **Software
qualification.** The numerical analysis is textbook. What was not known was that this repository's
reported K uncertainties were governed by it.

*What could belong in a thesis methodology chapter?* The qualification methodology: using exact
rational arithmetic as a covariance oracle for a small augmented system; separating weak-but-
resolvable from below-epsilon by a scale-invariant singular-value ratio; and the demonstration that
a finite covariance from a below-epsilon direction varies by 1.59× with summation order and is
therefore meaningless.

*What would a paper-level contribution require?* A systematic study across mission geometries and
parameter types establishing when floored normal-matrix covariance materially misleads, with a
recommended diagnostic. R1COV has one parameter, one geometry, one synthetic truth.

`ACADEMIC_LITERATURE_ALIGNMENT = CONSISTENT_WITH_ESTABLISHED_NUMERICAL_LEAST_SQUARES_THEORY`

## 35. Research Value

```
RESEARCH_VALUE_CLASS = SOFTWARE_QUALIFICATION_ONLY
```

Not inflated. The method is standard, the theory is established, and the scientific conclusion about
K_SRP is unchanged by the repair — R1M's finding is confirmed, not revised. What R1COV delivers is a
qualified covariance path and a correction to previously published uncertainties. The oracle
methodology in section 34 is the part with reuse value beyond this repository, and it is a methods
contribution rather than a result.

## 36. Industry / Operational Interpretation

*What mission decision could have been wrong?* Any acceptance of K_SRP as a delivered solve-for
parameter. A reported `sigma_K` of `1.6e-03` against a truth of `1e-02` reads as a 16%
determination — a parameter worth delivering. The true figure of `1.9e-02` is 194%, a parameter that
is not determined at all. Consider-covariance analyses, prediction envelopes and maneuver margins
built on the former would all have been optimistic.

*Would the old result produce overconfident delivered covariance?* Yes, by 10–260× on the tested
cases, always optimistically, and worse the weaker the parameter.

*How should weak solve-for parameters be reported operationally?* With a fractional uncertainty
alongside the sigma, and with an explicit numerical-observability diagnostic. A parameter at 2.2
sigma from zero should be reported as characterized, not determined.

*Should rank and covariance-method diagnostics be exposed?* Yes. The rank margin, the singular-value
ratio, and which covariance method produced the answer should all be available to the consumer.
This phase exposes the first two through `RankDeficientCovarianceError`.

*What validation is needed before formal covariance can be mission-qualified?* Extension of this
qualification to the default six-state paths, Monte Carlo consistency showing the formal covariance
matches empirical scatter, and coverage of geometries beyond the single synthetic campaign arc.

```
OPERATIONAL_COVARIANCE_QUALIFICATION_CLASS =
  QUALIFIED_FOR_K_SOLVE_FOR_PATHS_ONLY_NOT_MISSION_QUALIFIED
```

## 37. Other Experiments Available

Discussed, not executed (§54):

- **Extend the square-root path to the default six-state estimators.** Addresses whether historical
  six-state covariances are also floor-affected — currently unknown and not claimed either way.
- **Monte Carlo consistency.** Does the qualified formal covariance match empirical scatter? This is
  the test that would move covariance from "numerically correct" to "statistically validated".
- **Much longer spans (20–50 orbits)** where Sun geometry rotates substantially.
- **Eclipse-rich schedules**, to test R1M's degeneracy mechanism directly.
- **ΔDOR / VLBI angular, optical LOS, lunar landmark optical navigation, inter-satellite
  range/range-rate** — genuinely new information directions (R1O).
- **Piecewise / stochastic K**, a different parameterization rather than a different observable.
- **Truth-model mismatch**, since everything here is noise-free synthetic truth.

## 38. Why They Come Later

Any new-observable experiment that reported a `sigma_K` before this phase would have inherited the
floor artifact and could have produced a confident false conclusion — which is exactly what happened
to R1 and R1G. That is the necessity argument for ordering, and it is now discharged.

The remaining ordering constraint is the same one-layer-at-a-time principle: if a new observable and
a covariance reformulation were introduced together and K became determinable, the cause would be
unattributable.

## 39. Regression Protection

See the final field block for the targeted battery tally. All required gates `PASS`:
P21, Model-S, long-arc, event conditioning, force / trajectory / range / counted-Doppler /
measurement K sensitivity, derivative chain, BLS / SRIF / SR-UKF default parity, and the common
Gaussian posterior gate.

One sourcing note, so the evidence is not overstated. The targeted battery deliberately excluded
`tests/test_r2_measurement_fidelity.py`, because that file carries the two known pre-existing
non-pass classes and would have obscured whether R1COV introduced anything. The four **Model-S
oracle** tests live in that file, so `MODEL_S_REGRESSION` is evidenced by the **full suite** run in
section 40, not by the targeted battery. They pass there, and neither of that file's known failures
is a Model-S test.

19 new tests were added in `tests/test_k_srp_square_root_covariance.py`, covering agreement with the
exact rational oracle, symmetry and positive definiteness, `R^T R = information`, scale invariance
without a prior, prior response, semi-definite prior acceptance, weak-direction retention,
`1/weak` proportionality, rank-deficiency refusal, and characterization of both the floored path and
the default pseudoinverse so the repair cannot silently regress.

One existing docstring in `tests/test_k_srp_bls_solve_for.py` was updated. It documented the floor
limitation as something Phase 17-R could not fix; that is no longer true on the K path. No assertion
in that test was changed.

## 40. Full Suite

```
TOTAL_TESTS_COLLECTED = 1264
TESTS_PASSED          = 1223
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
```

The decisive comparison is against R1M's baseline on the same suite:

| | R1M baseline | R1COV | delta |
|---|---:|---:|---:|
| collected | 1245 | 1264 | **+19** |
| passed | 1204 | 1223 | **+19** |
| failed | 2 | 2 | 0 |
| errors | 10 | 10 | 0 |
| skipped | 29 | 29 | 0 |

The entire delta is the 19 new tests, all passing. The non-pass set is **identical** to R1M's, test
for test — the same two failures and the same ten errors, with no membership change. A production
change to a covariance path that altered anything else would have shown up here.

| non-pass | count | classification |
|---|---:|---|
| `test_r2_measurement_fidelity.py` module fixture — `NameError: CLOSURE_CURRENT_TREE_SHA256` | 10 errors | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change` — Phase 16 `dynamics.py` bytes | 1 failed | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_fa06_pytest_session_imports_isolated_repository` — root-`pytest.ini` injects the sibling worktree onto `pythonpath` | 1 failed | `KNOWN_PREEXISTING_ENVIRONMENT` |
| slow/optional regressions behind `LUNAR_OD_RUN_SLOW_TESTS=1` and optional dependencies | 29 skipped | `KNOWN_PREEXISTING_ENVIRONMENT` |

Neither known failure is a Model-S oracle test, so `MODEL_S_REGRESSION = PASS` on the four Model-S
tests in that file. None of these were repaired (§13, and §56 of R1M's governing spec).

```
KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1COV_INTRODUCED_NONPASSES              = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0
```

## 41. What This Phase Actually Established

1. The R1M defect reproduces on shipped code: 260.6× / 11.8× / 10.2× overconfidence.
2. `cond(H^T W H) = cond(H_w)^2` to eight digits, confirming the squaring as root cause.
3. An exact rational covariance oracle is achievable for this system, and five independent
   arithmetic routes agree with it.
4. A square-root covariance path reproduces the exact covariance to `≤1.6e-13`, is exactly
   scale-invariant, and responds correctly to priors.
5. The shipped `posterior_sqrt_information` was a second, previously unknown instance of the same
   defect (90–99.6% error).
6. A weak direction can be retained with a correctly large uncertainty; a below-epsilon direction's
   finite covariance is meaningless (1.59× spread with summation order) and is refused.
7. Default `pinv` fails in the opposite, more dangerous direction — eight orders of magnitude.
8. The repair is bitwise-safe for point estimates and for every default path.
9. K_SRP is **numerically observable but not practically identifiable**: best case 45.2% fractional
   uncertainty, 2.21 sigma.
10. R1M's scientific conclusion is confirmed on qualified covariance, not merely on a proxy.

## 42. What Remains Unknown

- Whether the **default six-state** covariances are also floor-affected. Not tested, not claimed.
  Those paths are well conditioned in ordinary use, but "well conditioned in ordinary use" is an
  expectation, not a measurement.
- Whether the qualified formal covariance is **statistically** correct — no Monte Carlo consistency
  check was run. This phase establishes numerical correctness against an exact oracle, which is a
  different claim.
- Whether the repair holds on geometries beyond the three arcs and the synthetic controls tested.
- Whether the K point estimate itself is biased in the weak regime. R1G saw solve-K collapse toward
  zero at 3 and 5 orbits; the live G1 solve here returned `K = 1.04e-03` against a truth of `1e-02`,
  well inside the qualified `sigma_K` of `1.94e-02` but far from truth. Covariance repair does not
  address estimator behaviour in a weak nonlinear regime, and that remains open.
- Everything is `SYNTHETIC_CAMPAIGN_TRUTH`, never `SPACECRAFT_TRUTH`.

## 43. Decision Tree From Here

Applying §68:

- *If square-root K covariance is qualified* → **satisfied.** Every gate passed, integration is
  complete, point estimates and default paths are provably untouched.
- *If the method works but integration is blocked by legacy contracts* → not applicable; integration
  succeeded within scope.
- *If square-root covariance cannot be qualified* → not applicable.

Therefore proceed to **Phase 17-R1O — Additional Observable Feasibility Study**, which can now
report formal uncertainties that mean something.

Two secondary items should be scheduled but not bundled into R1O: extending the square-root path to
the default six-state estimators, and a Monte Carlo consistency check of the qualified covariance.
Neither blocks R1O, because R1O's conclusions will rest on the now-qualified K path.

## 44. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = f0ee0fac0f07db8ad3da60ef7434749eaa5438a4
START_TREE   = c646b86d9281d4053155a686c80fa1e7bfbbf737

R1M_CONTENT_COMMIT_REACHABLE = YES   (f2af3045e44860b0892fe84b788ab7b75c7c63a2)

R1M_COVARIANCE_DEFECT_REPRODUCED = YES

G0_OLD_SIGMA_K    = 3.4688695056e-03
G0_ORACLE_SIGMA_K = 9.0401755679e-01
G0_QR_SIGMA_K     = 9.0401755679e-01   (rel 1.55e-13)
G0_SRIF_SIGMA_K   = 9.0401755679e-01   (rel 5.21e-14)

G1_OLD_SIGMA_K    = 1.6448102377e-03
G1_ORACLE_SIGMA_K = 1.9394564345e-02
G1_QR_SIGMA_K     = 1.9394564345e-02   (rel 7.03e-14)
G1_SRIF_SIGMA_K   = 1.9394564345e-02   (rel 1.39e-13)

HIGH_PRECISION_ORACLE_CONVERGENCE_GATE = PASS
NORMAL_MATRIX_CONDITION_SQUARING_GATE  = PASS

QR_COVARIANCE_ORACLE_GATE              = PASS
QR_COVARIANCE_SCALING_INVARIANCE_GATE  = PASS
QR_PRIOR_RESPONSE_GATE                 = PASS

SRIF_R_FACTOR_COVARIANCE_GATE          = PASS
BLS_SRIF_COVARIANCE_CONSISTENCY_GATE   = PASS

SRUKF_AFFECTED_BY_NORMAL_MATRIX_FLOOR  = NO

WEAK_DIRECTION_RETENTION_GATE          = PASS
TRUE_RANK_DEFICIENCY_SEMANTICS_GATE    = PASS

POINT_ESTIMATE_INVARIANCE_GATE         = PASS   (bitwise)

PRIOR_DIAGNOSTIC_POST_REPAIR_GATE      = PASS

G0_QUALIFIED_SIGMA_K = 9.040176e-01     (90.4x K_truth)
G1_QUALIFIED_SIGMA_K = 1.939456e-02     (1.94x K_truth)

BEST_EXISTING_RADIOMETRIC_SIGMA_K                  = 4.516318e-03
BEST_EXISTING_RADIOMETRIC_FRACTIONAL_K_UNCERTAINTY = 45.2%  (2.21 sigma)

ESTIMATOR_SIGMA_K_TRUST_STATUS_AFTER_R1COV = QUALIFIED
    (for the BLS and SRIF K solve-for paths only; the default six-state
     paths retain the floored helper and are NOT covered)

PRODUCTION_CODE_CHANGED          = YES   (lunar_od/estimators.py only)
POINT_ESTIMATE_BEHAVIOR_CHANGED  = NO
PHYSICAL_MODEL_CHANGED           = NO
MEASUREMENT_MODEL_CHANGED        = NO
FORMAL_COVARIANCE_METHOD_CHANGED = YES   (K solve-for paths only)
DEFAULT_6STATE_BEHAVIOR_CHANGED  = NO

RESEARCH_VALUE_CLASS = SOFTWARE_QUALIFICATION_ONLY
ACADEMIC_LITERATURE_ALIGNMENT =
  CONSISTENT_WITH_ESTABLISHED_NUMERICAL_LEAST_SQUARES_THEORY
OPERATIONAL_COVARIANCE_QUALIFICATION_CLASS =
  QUALIFIED_FOR_K_SOLVE_FOR_PATHS_ONLY_NOT_MISSION_QUALIFIED

P21_REGRESSION                           = PASS
MODEL_S_REGRESSION                       = PASS   (full suite; see section 39)
LONG_ARC_REGRESSION                      = PASS
EVENT_CONDITIONING_REGRESSION            = PASS

FORCE_K_DERIVATIVE_REGRESSION            = PASS
TRAJECTORY_K_SENSITIVITY_REGRESSION      = PASS
RANGE_K_SENSITIVITY_REGRESSION           = PASS
COUNTED_DOPPLER_K_SENSITIVITY_REGRESSION = PASS
MEASUREMENT_K_SENSITIVITY                = PASS
DERIVATIVE_CHAIN_GATE                    = PASS

BLS_DEFAULT_PARITY             = PASS
SRIF_DEFAULT_PARITY            = PASS
SRUKF_DEFAULT_PARITY           = PASS
COMMON_GAUSSIAN_POSTERIOR_GATE = PASS

TARGETED_BATTERY = 344 collected, 332 passed, 0 failed, 0 errors, 12 skipped

TOTAL_TESTS_COLLECTED = 1264
TESTS_PASSED          = 1223
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
    (R1M baseline 1245/1204/2/10/29; the entire delta is the 19 new tests,
     and the non-pass set is identical test for test)

KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1COV_INTRODUCED_NONPASSES              = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0

MAIN_CHANGED        = NO
ORIGIN_MAIN_CHANGED = NO

REPORT_COMPLETENESS_GATE = PASS
    all 45 mandatory sections present; §58 per-test format applied in full to
    section 7 and abbreviated (purpose / method / results / interpretation /
    verdict) elsewhere, with the shared setup stated once in section 7;
    all 9 §56 artifacts and all 6 §57 figures produced.

PHASE17_R1COV_GATE = PASS

PRIMARY_CLASS =
  K_SRP_FORMAL_COVARIANCE_REPAIRED_BY_SQUARE_ROOT_RECOVERY
  AND_K_CONFIRMED_NUMERICALLY_OBSERVABLE_BUT_NOT_PRACTICALLY_IDENTIFIABLE

NEXT_ACTION =
  PHASE_17_R1O_ADDITIONAL_OBSERVABLE_FEASIBILITY.
  Secondary, unbundled: extend the square-root path to the default
  six-state estimators; Monte Carlo consistency of the qualified
  covariance. Neither begins automatically.

COMMITS_CREATED = 1
COMMIT_LIST     = recorded in artifacts/r1cov_manifest.json
    (a commit cannot embed its own hash; the manifest written alongside this
     report records the parent, and the tip is reported in the phase closure
     message)

PUSH = NONE
MERGE = NONE
MAIN_MODIFICATION = NONE
HISTORY_REWRITE = NONE
FORCE_PUSH = NONE
```

## 45. Exact Next Action

**STOP** (§70). No R1O, no new measurement physics, no merge, no push, no change to main.

The phase's question:

> Can the production OD framework now report a numerically trustworthy formal uncertainty for weak
> K_SRP solve-for problems?

**Yes, for the two explicitly qualified paths.** BLS and SRIF K solve-for covariance is now
recovered from an orthogonal factorization of the design matrix, agrees with exact rational
arithmetic to `≤1.6e-13`, is exactly invariant to numerical scaling, responds correctly to priors,
retains weak directions with honestly large uncertainties, and refuses to invent a covariance for a
direction that carries none. The point estimates and every default path are bitwise unchanged.

Two qualifications must travel with that answer. The default six-state covariance path still uses
the floored helper and is **not** covered by this qualification. And numerical correctness against
an exact oracle is not statistical validation — a Monte Carlo consistency check has not been run, so
"mission-qualified" is not claimed.

The scientific consequence is that K_SRP's previously reported uncertainties were optimistic by
10–260×, and the corrected numbers confirm rather than overturn R1M: K_SRP is numerically observable
in this geometry and not practically identifiable from it.
