# PHASE 17-R1M — MULTI-ARC K_SRP IDENTIFIABILITY AND COVARIANCE-DIAGNOSTIC STUDY

## 1. Executive Summary

R1M asked whether observations from several separated tracking arcs, each with its own local
orbital state but sharing one physical `K_SRP`, can separate `K_SRP` from the local state
corrections. It also asked what actually produces the estimator-reported `sigma_K`, which R1G
could not reproduce and left as `PRIOR_DIAGNOSTIC_STILL_UNRESOLVED`.

Both questions are now answered, and the second answer is the more important one.

**The covariance question is resolved, and the answer is that the reported `sigma_K` is a
numerical artifact.** The estimator-reported `sigma_K` is not set by the data and not set by the
prior. It is set entirely by the eigenvalue safety floor inside `_safe_covariance_from_information`.
For G0, `sqrt(K_scale^2 / floor)` reproduces the reported value to `1.5e-10` relative; the floor
inflates the smallest eigenvalue by a factor of `67,917`. The true data-only `sigma_K` for G0 is
`0.904 m^2/kg`, not the reported `0.00347` — the reported figure is 260 times too optimistic on a
parameter whose truth value is `0.01`. Under a change of the K scaling constant, every physical
quantity is bitwise invariant (spread `0.00e+00`) while the reported `sigma_K` moves linearly. A
scale-dependent answer to a scale-free question is the definition of an artifact.

A second, independent numerical finding reinforces this. The rank deficiency itself is
manufactured by the normal-matrix formulation. The whitened, scaled **design** matrix is
conditioned at `2.6e9` (G0) and `1.0e8`–`1.2e8` (G1–G3), all comfortably resolvable in double
precision where `1/eps = 4.5e15`. Forming `H^T W H` squares that condition number exactly —
measured ratio `cond(info)/cond(Hs)^2 = 1.0000` in all four cases — producing `6.8e18` and pushing
the seventh direction below machine epsilon. The conditional K information is identical to `2e-9`
whether computed through the normal matrix or through a QR square-root route; only the **inverse**
is destroyed. The information is there. The covariance pipeline cannot see it.

**On identifiability, multi-arc processing accumulates information but never separates K.** The
ordering, measured on identical truth and identical measurement physics, is decisive:

| configuration | data used | `I(K|x)` |
|---|---:|---:|
| 3 separate 2.0-orbit arcs, pure multi-arc (F3, Model A) | 6.0 orbits, 386 obs | 5.54e+03 |
| 1 contiguous 5.0-orbit arc (G3) | 5.0 orbits, 307 obs | 3.52e+04 |
| 5 arcs linked across 5.5 orbits (M3, Model B) | 2.5 orbits, 295 obs | 4.90e+04 |

Pure multi-arc is **strictly worse than simply tracking contiguously for the same duration** — it
gets 6.4× less information from 26% more data. The configuration that does help, the
continuity-linked model, wins with 2.4× *less* data than pure multi-arc, because what buys K
information is elapsed dynamical span, not observation count.

Crucially, the continuity-linked model **is not a new architecture**. It is the existing
single-arc estimator fed a data set with gaps. It requires no production change whatsoever.

Yet even the best case falls short of practical identifiability. M3 Model B reaches a data-only
`sigma_K` of `4.52e-3` against a truth of `0.01` — 45% fractional uncertainty, about `2.2 sigma`.
Rank remains 6/7 in every one of the nineteen configurations tested, and the weakest mode remains
`0.99999999` aligned with K throughout. The orthogonal fraction `f_perp` saturates near `0.28`–`0.30`
and never approaches 1.

Therefore `MULTI_ARC_PRODUCTION_IMPLEMENTATION_JUSTIFIED = NO`, on two independent grounds: the
pure multi-arc architecture that would have to be built is strictly worse than what exists, and the
configuration that is better needs nothing built.

**One correction to my own provisional finding is recorded prominently in section 17.** The
pre-declared 0.5-orbit campaign showed Model A losing information by a factor of `1.3e6`, and that
number is real but it is not a property of pure multi-arc. It is a property of the window length I
chose. A fairness control with 2.0-orbit arcs collapses the penalty to 15–29×. The `1.3e6` figure
must not be quoted as a result about multi-arc formulations.

## 2. What R1/R1G Actually Established

R1 demonstrated that the production estimators can technically solve for a static `K_SRP`, and that
on the selected short single-station range campaign the result was weakly observable and not
predictively useful.

R1G tested whether arc length, station geometry, counted-Doppler information, or range+Doppler
combination could remove that limitation. None did. R1G established
`PRACTICAL_GEOMETRY_FOUND = NO` and `K_SRP_REMAINS_WEAKLY_OBSERVABLE_ACROSS_TESTED_GEOMETRIES`.

R1G's scientifically important observation, which R1M inherits and confirms, is that increasing
measurement sensitivity did not automatically create an independent K direction. Conditional K
information rose four orders of magnitude from 1.3 to 5 orbits while rank stayed 6/7 and the
weakest mode stayed almost entirely K.

R1G also left two items explicitly open: the prior/covariance diagnostic
(`PRIOR_DIAGNOSTIC_STILL_UNRESOLVED`), and the untested question of whether *separated* arcs
behave differently from one longer arc.

## 3. Why R1M Was the Next Rational Step

Extending a single contiguous arc further was already shown to be scientifically unjustified: the
3- and 5-orbit nonlinear solves became unstable and drove K toward its physical lower boundary.

The remaining structural hypothesis was that separated arcs might help for a reason longer arcs
cannot. In a multi-arc formulation each arc carries its own local state, so a K signature that a
single arc's state correction can absorb might not be absorbable by *all* arcs' corrections
simultaneously using one shared K. This is the standard argument for global-parameter estimation
in planetary radio science, and it had not been tested here.

Independently, no K uncertainty statement could be trusted at all until the covariance pipeline was
traced, because R1G had shown the reported `sigma_K` was not reproducible by any scalar calculation.
Running more identifiability experiments while the uncertainty metric was unexplained would have
risked building conclusions on an uninterpretable number. R1M therefore ran the covariance audit
first.

## 4. Academic Context

The multi-arc formulation with local states and shared global parameters is the standard
architecture in Milani & Gronchi's treatment of orbit determination, and is the operational basis
for gravity-field and parameter recovery in planetary radio science (Juno, GRAIL, MRO). In that
literature the global parameters recovered this way are typically ones with a *secular, arc-crossing*
signature — gravity harmonics, tidal Love numbers, ephemeris corrections.

R1M's result is consistent with that literature rather than contrary to it. Multi-arc works for
parameters whose signature accumulates across the span in a way local state corrections cannot
mimic. `K_SRP` on a low lunar orbit over 5–7 orbits is not such a parameter in the tested geometry:
its signature within any single short arc is nearly reproducible by that arc's own six-state
correction, and the pure multi-arc formulation adds one shared unknown while adding six local
unknowns per arc — a net loss of leverage.

The cislunar observability literature's distinction between *information accumulation* and
*information direction* is exactly the distinction R1M measures with the orthogonal fraction.

## 5. Industry / Mission-Operations Context

For a navigation or POD team three findings here are operationally relevant, in descending order of
importance.

First, a formal parameter sigma emitted by a covariance pipeline with an eigenvalue floor can be
silently floor-dominated and optimistic by orders of magnitude, in a way that looks entirely
plausible — G0's reported `0.00347` is a believable-looking number, and it is wrong by 260×. Any
operational use of formal `sigma_K` for consider-covariance, maneuver planning margin, or
delivery-accuracy prediction would inherit that error.

Second, the SVD pseudoinverse — a common defensive reflex for near-singular covariance work — is
the **worst** of the five methods tested here, not the safest. Default `rcond` discards the K
singular value, so K's variance contribution vanishes and the method reports `5.5e-10`: essentially
zero uncertainty on an essentially unconstrained parameter. This failure is silent and dangerous.

Third, if a team wants K observability from radiometrics, the operationally useful lever is
**elapsed span between passes, not tracking volume**. Spreading 2.5 orbits of tracking across a
5.5-orbit span beat 2.5 orbits tracked contiguously by 10.2×, and beat 6.0 orbits of tracking
arranged as separate 2-orbit arcs. Tracking time is the expensive resource; span is comparatively
cheap.

## 6. Entering Repository State

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 7acb10ac511ac01ffb0f2aed08034ed403e9bd51
START_TREE   = 68f2a1af1f9f271ade72ab6354071d79e4594bb7
```

Both R1G commits were verified reachable from `START_HEAD`:
`3abc8838ff3e99bd53389eb90d9f7def2173208b` and `7acb10ac511ac01ffb0f2aed08034ed403e9bd51`.

Canonical main was verified unchanged at `fea476f81dad07b3914e53eab709e10fa6e9d10b`, for both
`main` and `origin/main`. `R1M_INPUT_GATE = PASS`.

Throughout the phase `git status --porcelain --untracked-files=no` remained empty. No tracked file
was modified at any point. This is the basis for the regression classification in section 37: no
non-pass can have been introduced by R1M, because R1M changed no tracked byte.

## 7. G0/G1 Reproduction

**Purpose.** Confirm R1M operates on the same arc and same information structure as R1/R1G before
drawing any new conclusion.

**Scientific hypothesis.** The frozen production Jacobians, propagated through the same campaign
epoch and cadence, reproduce R1G's published information structure.

**Setup.** Campaign epoch and orbit period read from
`phase7_measurement_geometry_manifest.json`; initial state from `phase8_X_ext.npy` at `t=0`.
`K_truth = 0.01 m^2/kg` (`SYNTHETIC_CAMPAIGN_TRUTH`), cadence 60 s, integrator `rtol=1e-12`,
`atol=1e-13`, two-way converged-event range with `tolerance_s=1e-13`, noise-free, minimum elevation
10°, scaling `diag(1e6,1e6,1e6,1e3,1e3,1e3,1e-2)`.

**Method.** Read `H_x0` and the K column out of the frozen production code, form the scaled
information matrix, and compare rank, spectrum, weakest mode, raw `I_KK`, conditional K
information, and reported `sigma_K` against R1G's published table.

**What changed.** Nothing. Physics, measurement model, estimator and numerics are all the frozen
production code; only the analysis harness is new.

**Results.**

| quantity | R1G G0 | R1M G0 | R1G G1 | R1M G1 |
|---|---:|---:|---:|---:|
| rank | 6/7 | 6/7 | 6/7 | 6/7 |
| scaled condition | 6.79e18 | 6.791689e18 | 1.39e16 | 1.390364e16 |
| weakest-mode K component | 0.99999999985 | 0.9999999998485 | 0.99999999376 | 0.9999999937609 |
| smallest scaled singular value | 1.22e-4 | 1.223619e-4 | 2.66e-1 | 2.658520e-1 |
| raw `I_KK` | not published | 1.5500424e+04 | not published | 5.2687509e+04 |
| conditional K information | **1.5212** | **1.2236192** | 2658.51 | 2658.5201 |
| reported `sigma_K` | 0.00346885 | 0.0034688695 | 0.00164481 | 0.0016448102 |

Every structural quantity reproduces to nine or more significant figures. One entry does not: G0's
conditional K information, where R1G reports `1.5212` and R1M obtains `1.2236192`, a 19.6%
discrepancy.

This was investigated rather than absorbed. R1M's value was computed by six independent numerical
routes — exact `solve`-based Schur complement, `pinv` at several `rcond` values, `lstsq` on the
normal matrix, and a QR projector on the design matrix directly — which agree with each other in the
range `1.2236`–`1.2246`. R1G's `1.5212` was not reproducible by any of them.

The decisive evidence is internal to R1G. R1G's own G0 row lists the smallest scaled singular value
as `1.22e-4`. Because the weakest mode is `0.99999999985` aligned with K, that singular value
unscaled by `SCALE_K^2 = 1e-4` must approximately equal the conditional K information — giving
`1.2236`, R1M's value, not R1G's `1.5212`. Applying the identical check to G1 gives
`2.66e-1 / 1e-4 = 2660 ≈ 2658.5`, which *is* consistent with R1G's own G1 entry. R1G's G0
conditional-information figure is therefore internally inconsistent with R1G's own G0 singular
value, while R1M's is consistent with both.

The conclusion is that R1G's G0 conditional-information entry is in error and R1M's is correct. It
is recorded here rather than quietly overwritten. No R1M conclusion depends on which value is
right; G0 is unobservable under either.

**Two listed reproduction items were not re-run**: the nonlinear BLS K point estimate and the
holdout metrics. These are `NOT_RERUN_CARRIED_FORWARD_FROM_R1G`. They are not fabricated and not
silently omitted. The justification is that every R1M conclusion is drawn from information matrices
and their decompositions, never from a BLS point estimate or a holdout error; and that R1G's
reported `sigma_K` values, which *are* outputs of the nonlinear solve's covariance path, reproduce
here exactly (`0.0034688695` vs `0.00346885`; `0.0016448102` vs `0.00164481`), which confirms the
same solve configuration.

**Interpretation.** R1M is operating on the same arc, same physics, same information structure.

**System impact.** None.

**Academic interpretation.** Routine reproduction gate.

**Industry interpretation.** Standard configuration control before a follow-on study.

**Limitations.** K point estimate and holdout not independently re-run, as stated above.

**Verdict.** `PASS`. `G0_REPRODUCED = YES`, `G1_REPRODUCED = YES`.

## 8. Covariance Pipeline Audit

**Purpose.** Determine what actually sets the reported `sigma_K` — data, prior, or numerics.

**Scientific hypothesis.** If the reported `sigma_K` is set by the data, an independent scalar
Schur calculation must reproduce it. R1G showed it does not, so something else sets it.

**Setup.** As section 7. The audited path, unmodified, is:

```
design matrix  H = [H_x0 | h_K]      (frozen production Jacobians)
  -> physical scaling      Hs = H @ diag(1e6,1e6,1e6,1e3,1e3,1e3,1e-2)
  -> measurement weighting I  = Hs^T W Hs
  -> prior information     I += scale^T P0^-1 scale
  -> numerical inversion   eigh, then CLIP eigenvalues at max*1e-14
  -> unscaling             C  = scale C_scaled scale^T
  -> reported sigma_K      sqrt(C[6,6])
```

Of these steps, only the clipping is a numerical choice rather than a modelling one.

**Method.** Reconstruct `sigma_K` five independent ways on identical inputs.

**What changed.** Numerics only, in the analysis harness. The production covariance path was
executed, never modified.

**Results.**

| method | G0 | G1 |
|---|---:|---:|
| A. production estimator path (floored) | 3.4688695e-03 | 1.6448102e-03 |
| B. direct exact inverse | 9.0401755e-01 | 1.9394564e-02 |
| C. SVD pseudoinverse | 5.5425846e-10 | 3.7495863e-10 |
| D. data-only scalar Schur | 9.0401756e-01 | 1.9394564e-02 |
| E. full posterior, data + default state prior | 8.9531532e-01 | 1.9390722e-02 |

Methods B and D agree to `1.2e-8` (G0) and `1.0e-9` (G1), which is the expected agreement between an
exact matrix inverse and an independent scalar Schur complement. Method E differs from B by only
1.0% (G0) and 0.02% (G1), showing the default state prior is nearly irrelevant to K.

The production path (A) disagrees with all of them by 260× (G0) and 11.8× (G1), always in the
optimistic direction. The pseudoinverse (C) disagrees by nine orders of magnitude in the opposite
direction.

**Interpretation.** The reported `sigma_K` is neither the data answer nor the prior answer. It is
produced by a step that is neither.

**System impact.** None — no production behaviour changed.

**Academic interpretation.** A textbook demonstration that formal covariance from a near-singular
normal matrix is a property of the inversion algorithm as much as of the experiment.

**Industry interpretation.** Directly consequential; see section 5.

**Limitations.** Audited on G0 and G1 only; the mechanism is general but the magnitudes are
case-specific.

**Verdict.** `CHARACTERIZATION` — resolved in section 12.

## 9. Eigenvalue-Floor Audit

The floor is `max(max_eigenvalue * 1e-14, eps)`.

| | G0 | G1 |
|---|---:|---:|
| largest scaled eigenvalue | 8.310441e+14 | 3.696310e+15 |
| smallest scaled eigenvalue | 1.2236194e-04 | 2.6585200e-01 |
| floor | 8.3104411e+00 | 3.6963101e+01 |
| modes clipped | 1 of 7 | 1 of 7 |
| inflation of the clipped mode | **67,917×** | **139.0×** |
| K content of the clipped mode | 0.9999999998 | 0.9999999938 |
| `sqrt(K_scale^2 / floor)` | 3.4688695e-03 | 1.6448102e-03 |
| reported `sigma_K` | 3.4688695e-03 | 1.6448102e-03 |
| relative agreement | **1.5e-10** | **6.2e-09** |

Exactly one mode is clipped in each case, and that mode is essentially pure K. Predicting the
reported `sigma_K` from the floor alone — using no data quantity except the largest eigenvalue —
reproduces it to `1.5e-10`. There is no remaining room for a data contribution.

`G0_FLOOR_CONTROLS_K_VARIANCE = YES`, `G1_FLOOR_CONTROLS_K_VARIANCE = YES`.

**Verdict.** `PASS` (the audit succeeded; the finding is negative for the pipeline).

## 10. Scaling Audit

The K scaling constant is a bookkeeping choice. Physical results must not depend on it.

Three physically equivalent K scales were used: `5e-3`, `1e-2`, `2e-2`.

| quantity | G0 spread | G1 spread |
|---|---:|---:|
| conditional K information `I(K|x)` | 0.00e+00 | 0.00e+00 |
| exact-inverse `sigma_K` | 0.00e+00 | 0.00e+00 |
| **estimator-reported `sigma_K`** | **linear in K_scale** | **linear in K_scale** |

The physical quantities are bitwise invariant. The estimator-reported `sigma_K` scales linearly
with an arbitrary bookkeeping constant.

This is the cleanest single proof in the phase. The floor is an **absolute** threshold
(`max_eig * 1e-14`) applied to a matrix whose entries depend on the chosen scaling, so the clipped
value — and hence the reported variance — inherits that scaling. A scale-free physical question
cannot have a scale-dependent answer.

It also retroactively explains an R1G puzzle: why the broad prior (`sigma = 1.0`) and the moderate
prior (`sigma = 0.01`) produced *identical* reported `sigma_K`. Neither prior was being used. The
floor set the answer in both cases.

`SCALING_PHYSICAL_INVARIANCE_GATE = PASS`.

## 11. Rank-Tolerance Audit

Rank was evaluated across five justified relative tolerances. No tolerance was tuned, and none was
selected to produce a preferred answer (§21).

| relative tolerance | G0 rank | G1 rank |
|---|---:|---:|
| 1e-10 | 6/7 | 6/7 |
| 1e-12 | 6/7 | 6/7 |
| 1e-14 | 6/7 | 6/7 |
| 1e-15 | 6/7 | 6/7 |
| 1e-16 | 6/7 | 6/7 |

Smallest/largest singular ratio: `1.472e-19` (G0), `7.192e-17` (G1). Both are below double epsilon
(`2.22e-16`), so the rank is **not** tolerance-dependent — there is no defensible cut anywhere in
the tested range that yields 7/7. The complete spectra are published in
`r1m_multi_arc_singular_values.csv` so the reader can verify this independently.

**But the near-singularity is manufactured by the formulation, not by the data.** Measuring the
whitened, scaled *design* matrix directly:

| case | `cond(Hs)` | `cond(info)` | `cond(info)/cond(Hs)^2` | design resolvable? | info resolvable? |
|---|---:|---:|---:|:--:|:--:|
| G0 | 2.6061e+09 | 6.7917e+18 | 1.0000 | yes | no |
| G1 | 1.1791e+08 | 1.3904e+16 | 1.0000 | yes | no |
| G2 | 1.1092e+08 | 1.2304e+16 | 1.0000 | yes | no |
| G3 | 1.0171e+08 | 1.0345e+16 | 1.0000 | yes | no |

The information matrix's condition number is the design matrix's squared, to a measured ratio of
exactly `1.0000` in all four cases. The design matrix sits at `1e8`–`2.6e9` against a double-precision
limit of `1/eps = 4.5e15` — seven orders of margin. The normal matrix sits beyond that limit.

Confirming that the information itself survives while only the inverse is lost: the conditional K
information computed through the normal matrix agrees with the QR square-root route to `2.7e-8`
(G0) and `2.0e-9` (G1–G3).

`RANK_TOLERANCE_CLASS = NUMERICALLY_NEAR_SINGULAR` in the audited normal-matrix pipeline.
In the square-root domain the same seventh direction is `WEAK_BUT_NONZERO` and resolvable with
large margin. Both statements are true of different formulations of the same data, and reporting
only the first would misattribute a software property to physics.

## 12. Prior Diagnostic Resolution

R1G left this `PRIOR_DIAGNOSTIC_STILL_UNRESOLVED`. It is now resolved.

The reported `sigma_K` is not data-limited and not prior-driven. It is produced by the eigenvalue
safety floor, demonstrated three independent ways: the floor alone predicts it to `1.5e-10`; it
scales linearly with an arbitrary bookkeeping constant while every physical quantity is bitwise
invariant; and it is reproduced by neither of the two methods that agree with each other.

```
PRIOR_DIAGNOSTIC_STATUS = RESOLVED_NUMERICAL_FLOOR_EFFECT
ESTIMATOR_SIGMA_K_TRUST_STATUS = NOT_RELIABLE_IN_NEAR_SINGULAR_REGIME
COVARIANCE_DIAGNOSTIC_GATE = PASS
```

The gate is `PASS` because the diagnostic succeeded — the question is answered. The answer is that
the metric it examined must not be used.

For all subsequent sections, K uncertainty is reported as the data-only Schur value
`sigma_K = 1/sqrt(I(K|x))`, which is scale-invariant, reproducible by independent routes, and
free of the floor.

## 13. Single-Arc K/State Geometry

**Purpose.** Measure directly how much of K's measurement signature the six-state correction can
reproduce.

**Method.** In whitened coordinates with `A_x = W^(1/2) H_x0` and `b_K = W^(1/2) h_K`, split `b_K`
into the part reproducible by the state columns and the part that is not, using a QR projector
rather than forming `(A^T A)^-1` — squaring a badly conditioned matrix to answer a question *about*
conditioning would be self-defeating. The identity used throughout is

```
I(K|x) = I_KK - I_Kx I_xx^-1 I_xK = ||b_K - P_Ax b_K||^2 = ||b_K,perp||^2
f_perp = ||b_K,perp|| / ||b_K|| = sqrt( I(K|x) / I_KK )
```

so `f_perp` is not an independent quantity but the same fact as a dimensionless fraction.

**Results.**

| case | arc | obs | `I_KK` | `I(K|x)` | `f_perp` | rank | weakest-mode K | data-only `sigma_K` | `sigma_K / K_truth` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 1.3 orbit | 82 | 1.550042e+04 | 1.223619e+00 | 0.008885 | 6/7 | 0.999999999849 | 9.040e-01 | 90.40 |
| G1 | 2.0 orbits | 144 | 5.268751e+04 | 2.658520e+03 | 0.224629 | 6/7 | 0.999999993761 | 1.939e-02 | 1.94 |
| G2 | 3.0 orbits | 215 | 1.578665e+05 | 9.355313e+03 | 0.243436 | 6/7 | 0.999999994508 | 1.034e-02 | 1.03 |
| G3 | 5.0 orbits | 307 | 4.863287e+05 | 3.520275e+04 | 0.269044 | 6/7 | 0.999999995544 | 5.330e-03 | 0.53 |

**Interpretation.** The fractional uncertainty column is the operationally meaningful one and it
has not been reported before. Even G3, the longest contiguous arc, leaves 53% uncertainty on K. G0
leaves 9040%. No single arc comes close to a confident K determination.

**Verdict.** `CHARACTERIZATION`.

## 14. Orthogonal K Fraction

`f_perp` rises steeply from 1.3 to 2.0 orbits (0.0089 → 0.2246, a factor of 25) and then
**saturates**: 0.2246 → 0.2434 → 0.2690 across 2, 3 and 5 orbits. Going from 2 to 5 orbits — more
than doubling the arc — buys only a 20% relative increase in the independent fraction.

This is the central geometric fact of the whole K_SRP investigation. Roughly 73–99% of K's
measurement signature is reproducible by an adjustment of the initial orbital state, and that
proportion is close to a property of the dynamics rather than of the data volume. Accumulating more
data raises `I_KK` and `I(K|x)` together, in nearly fixed ratio, without making K more *distinct*.

```
G0_K_ORTHOGONAL_FRACTION = 0.008885
G1_K_ORTHOGONAL_FRACTION = 0.224629
G2_K_ORTHOGONAL_FRACTION = 0.243436
G3_K_ORTHOGONAL_FRACTION = 0.269044
SINGLE_ARC_IDENTIFIABILITY_CLASS = WEAK_BUT_NONZERO_AND_SATURATING
```

## 15. Weakest-Mode Physical Interpretation

Across all nineteen configurations, the weakest mode's K component stays in
`[0.99999999, 1.00000000]`. The least-determined direction of the estimation problem is, to eight
or nine decimal places, "change K and leave the state alone."

Physically: a cannonball SRP acceleration over a few lunar orbits produces a trajectory
perturbation that is nearly indistinguishable from a small change in the initial state vector. The
Sun direction rotates only slowly over 5–7 orbits, so the SRP acceleration is nearly constant in
inertial space, and a nearly constant acceleration integrates into position and velocity offsets —
precisely what the six-state correction is free to absorb. Breaking that degeneracy requires either
a much longer span over which the Sun geometry rotates substantially, or eclipse entries and exits
that impose a signature no smooth state correction can mimic, or an observable sensitive to
something other than the same range history.

This is why `f_perp` saturates and why more of the same measurement does not help.

## 16. Pure Multi-Arc Mathematical Model

Model A assembles the `(6M+1)` global information matrix with one local six-state per arc and one
shared K as the final unknown. The row block for arc `i` is

```
[ 0 ... A_i ... 0 | b_i ]
```

so the design matrix is block-diagonal in the state columns and dense only in the shared K column.
The scaling is `diag(1e6,1e6,1e6,1e3,1e3,1e3)` repeated `M` times, then `1e-2` for K.

Model B, the continuity-linked alternative, is built by propagating **one** trajectory from the
campaign epoch and masking observations to the same windows. Because the trajectory is never
re-initialised, `Phi` and `dx/dK` accumulate through the unobserved gaps. This is what makes the
comparison meaningful: both models see byte-identical observations; they differ only in whether the
state is reset at each arc boundary.

All arcs in both models are windows on a single physically continuous truth trajectory. Arcs
differ by which observations are used, never by a different truth.

**Schedule (declared before any information result was inspected, §29).** Window length 0.5 orbit,
with starts anchored on measured DSN pass structure rather than round numbers:

```
Canberra  0.48-1.09, 1.49-2.09, 2.49-3.09, 3.49-4.09, 4.49-4.56 orbits
Madrid    4.93-5.09, 5.49-6.10, 6.50-7.10, 7.50-8.09
Goldstone 8.55-9.00
```

| case | arcs | starts | span | reach |
|---|---:|---|---:|---:|
| M1 | 2 | 0.50, 2.50 | 2.50 | 3.00 |
| M2 | 3 | 0.50, 2.50, 5.50 | 5.50 | 6.00 |
| M3 | 5 | 0.50, 1.50, 2.50, 3.50, 5.50 | 5.50 | 6.00 |

"Span" (last window end minus *first window start*) is what `elapsed_orbits` means throughout;
"reach" is the outer propagation bound. All three complexes are enabled in both models so that
handover happens naturally and neither model is silently given different data.

An earlier draft used evenly spaced starts at 0/2/4/6/8 orbits. It was **discarded, not tuned**:
Canberra stops seeing the spacecraft after ~4.56 orbits, so the 6.0 and 8.0 windows contained no
data at all and M2/M3 silently collapsed onto the same result. The replacement anchors come from
the visibility structure alone and were not tuned against any K singular value.

## 17. Two-Arc Study

**Purpose.** Test whether two separated arcs sharing one K separate K from the local states.

**Results, M1 (2 arcs, 0.5-orbit windows, 2.50-orbit span, 118 observations):**

| | unknowns | rank | `I_KK` | `I(K|x)` | `f_perp` | weakest-mode K | data-only `sigma_K` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A pure | 13 | 12/13 | 4.125057e+01 | 2.628563e-03 | 0.007983 | 0.9999999998 | 1.950e+01 |
| Model B linked | 7 | 6/7 | 8.756330e+04 | 3.478595e+03 | 0.199315 | 0.9999999923 | 1.696e-02 |

Model B carries `1.32e6` times more conditional K information than Model A on identical
observations.

### The fairness control — a correction to my own provisional finding

That `1.32e6` factor is arithmetically correct and it is **not a valid result about pure
multi-arc**. It is dominated by my choice of a 0.5-orbit window.

The mechanism is `S_K(t_0) = 0`: the K sensitivity is reset to zero at the start of every arc in
Model A. Half an orbit is not enough time for it to grow, so Model A's per-arc contributions are
tiny and the Schur sum rule (section 20) faithfully sums tiny numbers. The comparison was
structurally unfair to Model A.

I therefore re-ran both models with 2.0-orbit windows — the length at which a *single* arc was
R1G's best stable case — with starts spaced so windows never share an endpoint:

| case | arcs | window | span | obs | Model A `I(K|x)` | Model B `I(K|x)` | B/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 2 | 0.5 orbit | 2.50 | 118 | 2.629e-03 | 3.479e+03 | **1.3e6×** |
| F2 | 2 | 2.0 orbits | 4.50 | 242 | 2.754e+03 | 4.032e+04 | **14.6×** |
| M2 | 3 | 0.5 orbit | 5.50 | 177 | 4.006e-03 | 3.659e+04 | **9.1e6×** |
| F3 | 3 | 2.0 orbits | 7.00 | 386 | 5.536e+03 | 1.580e+05 | **28.6×** |

The penalty for the pure multi-arc formulation is **15–29×, not 10^6×**. Model A's `f_perp` also
behaves differently once the arcs are adequate: it rises with arc count (0.2056 → 0.2573) where at
0.5-orbit windows it was pinned flat at 0.0080.

The `1.3e6` figure must not be quoted as a property of multi-arc formulations. It is a property of
short arcs. This correction is recorded because the uncorrected number would have been the most
quotable result in the phase and it would have been wrong.

**Verdict.** `CHARACTERIZATION`.

## 18. Three-Arc Study

**M2 (3 arcs, 0.5-orbit windows, 5.50-orbit span, 177 observations):**

| | unknowns | rank | `I_KK` | `I(K|x)` | `f_perp` | weakest-mode K | data-only `sigma_K` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A pure | 19 | 18/19 | 6.200558e+01 | 4.006460e-03 | 0.008038 | 0.9999999997 | 1.580e+01 |
| Model B linked | 7 | 6/7 | 4.057636e+05 | 3.658760e+04 | 0.300283 | 0.9999999970 | 5.228e-03 |

Adding a third arc raises Model A's `I(K|x)` by 1.52× while `f_perp` moves from 0.007983 to
0.008038 — a 0.7% change. Model B gains 10.5× and `f_perp` moves 0.199 → 0.300, a genuine 51%
increase in the independent fraction.

**Verdict.** `CHARACTERIZATION`.

## 19. Five-Arc Study

**M3 (5 arcs, 0.5-orbit windows, 5.50-orbit span, 295 observations):**

| | unknowns | rank | `I_KK` | `I(K|x)` | `f_perp` | weakest-mode K | data-only `sigma_K` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A pure | 31 | 30/31 | 1.018898e+02 | 6.477955e-03 | 0.007974 | 0.9999999995 | 1.242e+01 |
| Model B linked | 7 | 6/7 | 5.830936e+05 | 4.902650e+04 | 0.289965 | 0.9999999985 | 4.516e-03 |

Model A's `f_perp` across 2, 3 and 5 arcs is `0.007983`, `0.008038`, `0.007974` — **flat to within
0.8%, and not monotonic**. This is the signature the Schur sum rule predicts analytically
(section 20): pure multi-arc can only *accumulate magnitude*, never *rotate* K out of the local
state subspace. Every arc adds its own six local unknowns alongside its contribution, so the
proportion of K's signature that is independent stays put.

Model B's `f_perp` at 5 arcs (0.2900) is slightly *below* its 3-arc value (0.3003) because M2 and
M3 share the same 5.50-orbit span; M3 adds observations inside that span but no additional elapsed
dynamics. This is direct evidence, within Model B, that span rather than observation count is what
moves the geometry.

M3 Model B is the best configuration in the entire study: `sigma_K = 4.516e-03` against
`K_truth = 0.01`, i.e. **45% fractional uncertainty, about 2.2 sigma**. Rank is still 6/7.

**Verdict.** `CHARACTERIZATION`.

## 20. Multi-Arc Schur Sum Rule

**Scientific hypothesis.** If local states are genuinely independent across arcs, the global
conditional K information must equal the sum of the per-arc conditional K informations. This is a
falsifiable check on the assembly, and it also bounds what pure multi-arc can ever achieve.

| case | arcs | sum of per-arc `I(K|x)` | global Model A `I(K|x)` | relative difference | result |
|---|---:|---:|---:|---:|:--:|
| M1 | 2 | 2.62856176e-03 | 2.62856289e-03 | 4.31e-07 | PASS |
| M2 | 3 | 4.00645969e-03 | 4.00645999e-03 | 7.82e-08 | PASS |
| M3 | 5 | 6.47795217e-03 | 6.47795484e-03 | 4.12e-07 | PASS |

`MULTI_ARC_SCHUR_SUM_RULE = PASS`.

This does two jobs. It verifies the Model A assembly is correct — a mis-assembled block-diagonal
system would not satisfy it. And it proves the ceiling: because the global value is *exactly* the
sum of the parts, pure multi-arc cannot produce information that is not already present arc by arc.
It cannot create a new K direction; it can only add up existing ones.

The fairness control (section 17) is the same statement read forwards: since the total is the sum of
the per-arc values, the only way to make pure multi-arc competitive is to make each individual arc
informative — which means making the arcs longer, which means returning to the contiguous-arc
regime whose limits section 13 already measured.

**Verdict.** `PASS`.

## 21. Continuity-Linked Multi-Pass Model

Model B keeps one state at the campaign epoch and lets `Phi` and `dx/dK` accumulate through the
unobserved gaps. Its advantage over Model A has a clean physical reading: the gaps are not wasted.
Dynamics continue to propagate the K signature during them, so an observation at 5.5 orbits
constrains K using the *entire* 5.5 orbits of accumulated sensitivity, not merely the half-orbit
window it sits in.

The elapsed-time control isolates this from data volume by comparing against a contiguous arc
carrying the **same total observation time** but no gaps:

| case | contiguous | obs | contiguous `I(K|x)` | linked span | obs | linked `I(K|x)` | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 1.0 orbit | 72 | 5.337496e-01 | 2.50 orbits | 118 | 3.478595e+03 | **6517×** |
| M2 | 1.5 orbits | 85 | 3.364757e+02 | 5.50 orbits | 177 | 3.658760e+04 | **108.7×** |
| M3 | 2.5 orbits | 156 | 4.793520e+03 | 5.50 orbits | 295 | 4.902650e+04 | **10.2×** |

The same tracking time, spread over a longer elapsed span, is worth 10× to 6500× more K
information. The ratio shrinks as the contiguous control itself lengthens, consistent with
section 14's saturation.

**Verdict.** `CHARACTERIZATION`.

## 22. Pure-vs-Linked Comparison

The decisive comparison is not Model A against Model B at matched observations. It is Model A
against **what already exists**: contiguous single-arc tracking.

| configuration | tracking used | obs | `I(K|x)` | data-only `sigma_K` |
|---|---:|---:|---:|---:|
| F3 — 3 separate 2.0-orbit arcs, pure multi-arc | 6.0 orbits | 386 | 5.536e+03 | 1.344e-02 |
| G3 — 1 contiguous 5.0-orbit arc | 5.0 orbits | 307 | 3.520e+04 | 5.330e-03 |
| M3 — 5 arcs linked across 5.5 orbits | 2.5 orbits | 295 | 4.903e+04 | 4.516e-03 |

Pure multi-arc, given **more** tracking time and **more** observations than the existing contiguous
approach, delivers **6.4× less** conditional K information. The continuity-linked configuration
delivers the most, using 2.4× less tracking than the pure multi-arc case.

```
MODEL_A_INDEPENDENT_ARCS_CLASS = STRICTLY_WORSE_THAN_EXISTING_CONTIGUOUS_ARC
MODEL_B_CONTINUITY_LINKED_CLASS = BEST_TESTED_BUT_STILL_NOT_PRACTICALLY_IDENTIFIABLE
```

The operationally critical observation: **Model B is not a new architecture.** It is the existing
single-arc estimator applied to a data set with gaps — one state at one epoch, continuous
propagation, observations wherever they happen to exist. Production already supports it. The
configuration that helps requires no implementation; the configuration that would require
implementation does not help.

## 23. Optional Doppler / Range+Doppler Characterization

`NOT_RUN_BY_GATE` (§40 optional, must not block; §39 range first).

R1G's characterization is carried forward, not re-derived: counted-Doppler-only gave rank 6/7,
scaled condition `3.82e17`, conditional K information `247.5`; adding qualified two-way range rows
raised it only to `260.8` at rank 6/7. A 5.4% increase from adding an entire second measurement
family is consistent with section 15's explanation — both families observe the same range history,
so both are absorbed by the same state correction.

No geometric instantaneous range-rate was substituted for the qualified counted-Doppler observable
at any point in R1M (§41).

## 24. Multi-Arc Information Feasibility Verdict

```
MULTI_ARC_INFORMATION_CLASS = INFORMATION_ACCUMULATION_ONLY
```

**Numerically.** Conditional K information spans `2.6e-03` to `4.9e+04` across the tested
configurations — more than seven orders of magnitude. Over that entire range, rank is 6/7 (or
`6M/6M+1`) in every single case, the weakest-mode K component never leaves
`[0.99999999, 1.00000000]`, and `f_perp` never exceeds `0.3003`. The best data-only `sigma_K` is
45% of the truth value.

**Physically.** Information accumulates because SRP trajectory sensitivity grows with elapsed time.
It does not *separate* because, as section 15 explains, a nearly constant inertial acceleration over
a few orbits integrates into something a six-state correction can almost entirely reproduce. Adding
arcs adds both information and unknowns; adding span adds information without unknowns, which is why
Model B beats Model A — but span alone cannot rotate K out of a subspace it geometrically lies
nearly within.

```
MULTI_ARC_PRODUCTION_IMPLEMENTATION_JUSTIFIED = NO
```

Against the §43 criteria, the multi-arc *information gain* is in fact physical, stable under
scaling, not floor-produced, not prior-driven, and not the product of rank-tolerance manipulation.
Those necessary conditions are met. The verdict is nonetheless NO, on two independent sufficient
grounds:

1. The architecture that would be built — pure multi-arc with local states and shared K (Model A) —
   is **strictly worse** than the contiguous single-arc processing that already exists (section 22).
   Building it would reduce capability.
2. The configuration that *is* better (Model B) **requires no production change at all**. It is the
   existing estimator with gappy data.

Implementing a production multi-arc shared-K estimator would therefore spend significant effort to
obtain a worse result than is available today.

## 25. Optional Analysis Prototype

`NOT_RUN_BY_GATE`. §44 authorizes the analysis-only batch prototype only if
`MULTI_ARC_PRODUCTION_IMPLEMENTATION_JUSTIFIED = YES`. It is NO.

## 26. Optional Gaussian Oracle

`NOT_RUN_BY_GATE`. `MULTI_ARC_LINEAR_GAUSSIAN_ORACLE = NOT_RUN_BY_GATE`.

## 27. Optional Shared-K Recovery

`NOT_RUN_BY_GATE`. `MULTI_ARC_NOISE_FREE_RECOVERY = NOT_RUN_BY_GATE`.

## 28. Optional Future Holdout

`NOT_RUN_BY_GATE`. `MULTI_ARC_HOLDOUT_GATE = NOT_RUN_BY_GATE`. `artifacts/r1m_holdout.csv` is
correctly absent under §53's instruction to generate prototype artifacts only if the information
gate authorizes them.

## 29. What Changed in the System

```
PRODUCTION_CODE_CHANGED        = NO
PHYSICAL_MODEL_CHANGED         = NO
MEASUREMENT_MODEL_CHANGED      = NO
ESTIMATOR_ARCHITECTURE_CHANGED = NO
ANALYSIS_CAPABILITY_ADDED      = YES
```

The analysis capability added is five scripts under `examples/`, none of which is importable by or
reachable from production code:

- `phase17_r1m_core.py` — shared arc fixtures and information-geometry utilities
- `phase17_r1m_covariance_diagnostic.py` — the five-way covariance reconstruction and floor audit
- `phase17_r1m_multi_arc_study.py` — Model A / Model B assembly and the elapsed-time control
- `phase17_r1m_addendum.py` — fairness control, data-only `sigma_K`, figures
- `phase17_r1m_config_and_spectra.py` — configuration record and full singular spectra
- `phase17_r1m_figures2.py` — remaining required figures

These are analysis scripts. They are **not** production capability. Nothing in the shipped
estimator, dynamics, measurement or filter code can call them, and no production behaviour differs
by a single bit as a result of this phase.

## 30. What Did NOT Change

The frozen production model was not modified: `dynamics.py`, `two_way_range.py`, `radiometrics.py`,
`two_way_counted_doppler_reference.py`, `estimators.py`, `filters.py` were read and executed only.

No production multi-arc API was created. No BLS/SRIF/SR-UKF interface changed. No new measurement
physics was introduced. No new solve-for parameterization was added. No rank tolerance was altered
to manufacture 7/7. No geometric range-rate was substituted for counted Doppler. The historical
`wip/phase17-r-estimator-integration` branch was neither merged nor used as scientific truth.

`git status --porcelain --untracked-files=no` was empty for the entire phase.

## 31. Scientific Interpretation

Three findings, in descending order of evidential strength.

**First, and most robust: the reported `sigma_K` is a numerical artifact.** Reproduced to `1.5e-10`
from the floor alone, proven by a bitwise-invariance control, and consistent across two independent
cases. This is not a modelling judgement; it is a measurement of the software.

**Second: the rank deficiency is a formulation artifact.** `cond(info) = cond(Hs)^2` to a measured
ratio of exactly 1.0000. The design matrix is resolvable with seven orders of margin; the normal
matrix is not. The conditional information survives the normal-matrix route (agreeing to `2e-9`);
only the inverse is destroyed. The seventh direction is weak but genuinely nonzero in the data.

**Third, and the answer to the phase's question: multi-arc processing cannot make `K_SRP`
practically identifiable from the radiometric information we already trust.** The Schur sum rule
proves pure multi-arc can only accumulate, never rotate. The measured `f_perp` saturation shows that
what limits K is a geometric near-degeneracy between a slowly-rotating SRP acceleration and the
initial-state correction space — a property of the dynamics over a few lunar orbits, not of the
estimator, the measurement, or the arc arrangement.

The three findings interact in a way worth stating plainly. Findings one and two mean that *every
previously reported K uncertainty in this project, including R1's and R1G's, was floor-dominated and
optimistic.* Finding three means that fixing findings one and two will not rescue K observability —
it will make the true uncertainty visible, and the true uncertainty is worse than what was reported.
Honest numbers here move in the unhelpful direction, which is exactly why they need to be recorded
before any further K science is attempted.

## 32. Academic Counterpart and Literature Comparison

R1M reproduces, in a lunar SRP setting, the standard result that multi-arc global-parameter
estimation helps for parameters with arc-crossing secular signatures and does not help for
parameters absorbable within each arc's local state. The Schur sum rule is the formal expression of
that boundary, and measuring it directly — rather than inferring it from solve behaviour — is the
methodological contribution here.

The covariance finding has a direct literature counterpart in the long-standing preference for
square-root formulations (Potter, Bierman; SRIF as deployed in JPL navigation) precisely because
forming `H^T W H` squares the condition number. R1M provides a clean quantitative instance:
`cond(info)/cond(Hs)^2 = 1.0000` measured, with the design matrix resolvable and the normal matrix
not. That this project already has a qualified SRIF makes the remedy unusually concrete.

`ACADEMIC_LITERATURE_ALIGNMENT = CONSISTENT_WITH_ESTABLISHED_OD_THEORY`

The result confirms known theory in a specific new application. The floor-artifact diagnosis is a
software-qualification finding rather than a new scientific result.

`POTENTIAL_NOVELTY_CLASS = APPLICATION_SPECIFIC_METHODOLOGICAL_CONTRIBUTION`

## 33. Research / Thesis / Paper Value

```
RESEARCH_VALUE_CLASS = APPLICATION_SPECIFIC_RESEARCH_RESULT
```

Not inflated to a paper contribution. What R1M has is a careful, well-instrumented negative result
plus a software-qualification finding, in a setting where the underlying theory is established.

The defensible thesis content is the *method*: using the orthogonal decomposition `f_perp` and the
Schur sum rule to distinguish information accumulation from information rotation, and demonstrating
that a formal parameter sigma can be floor-dominated in a way that survives casual inspection. The
fairness control in section 17 is also instructive as a worked example of a plausible headline
number that does not survive its own control.

**Evidence still missing** for anything stronger: no nonlinear multi-arc recovery was run; no
holdout prediction was performed in R1M; eclipse-entry counts and Sun-geometry rotation were not
instrumented, so section 15's physical explanation of the saturation is well-motivated but not
directly measured; and the study covers one orbit geometry, one spacecraft configuration, and one
synthetic truth.

## 34. Industry / Operational Interpretation

```
OPERATIONAL_JUSTIFICATION_CLASS = NOT_JUSTIFIED_FOR_PRODUCTION_INTEGRATION
```

A navigation team reading this should take away: do not build the multi-arc estimator for this
purpose; it would be worse than what you have. If you want K observability from existing
radiometrics, buy elapsed span rather than tracking volume — and even then expect ~45% formal
uncertainty at best, which is not sufficient to use K as a delivered product. And do not trust a
formal parameter sigma from a floored normal-matrix inversion without auditing it the way section 8
does.

## 35. Other Experiments We Could Run

- Square-root (QR/SRIF) covariance path for K, replacing eigen-decomposition of `H^T W H`. Highest
  value, lowest risk: the remedy is already qualified in this codebase.
- Much longer spans (20–50 orbits) where Sun-direction rotation becomes substantial.
- Eclipse-rich geometries, instrumenting entry/exit counts, to test section 15's mechanism directly.
- Stochastic / piecewise-constant K instead of a single static K.
- Additional observables: ΔDOR / VLBI-like angular, optical line-of-sight, lunar landmark optical
  navigation, inter-satellite ranging.
- Nonlinear multi-arc recovery and holdout, if a future gate authorizes it.

## 36. Why We Are Not Running All of Them Yet

If multi-arc, a covariance reformulation, new observables, stochastic K and a longer span were
introduced together and K became observable, the result would be uninterpretable — we could not say
which mechanism did it, and therefore could not generalize, cost, or defend the finding. The
research must change one conceptual layer at a time.

There is also a strict ordering constraint specific to this phase's outcome. Sections 8–12
establish that the current formal `sigma_K` cannot be trusted. Any further K observability
experiment that reports a formal sigma would be reporting a floor artifact. **The covariance
qualification must come before the next observability experiment**, not in parallel with it, or the
next experiment's headline number will be as untrustworthy as R1's and R1G's were.

## 37. Regression Protection

Targeted battery, 12 files, `328 collected → 316 passed, 1 failed, 10 errors, 1 skipped`.

| gate | result |
|---|---|
| `P21_REGRESSION` | PASS |
| `MODEL_S_REGRESSION` | PASS (4 Model-S oracle tests, none among the non-passes) |
| `LONG_ARC_REGRESSION` | PASS |
| `EVENT_CONDITIONING_REGRESSION` | PASS |
| `FORCE_K_DERIVATIVE_REGRESSION` | PASS |
| `TRAJECTORY_K_SENSITIVITY_REGRESSION` | PASS |
| `RANGE_K_SENSITIVITY_REGRESSION` | PASS |
| `COUNTED_DOPPLER_K_SENSITIVITY_REGRESSION` | PASS |
| `MEASUREMENT_K_SENSITIVITY` | PASS |
| `DERIVATIVE_CHAIN_GATE` | PASS |
| `BLS_DEFAULT_PARITY` | PASS |
| `SRIF_DEFAULT_PARITY` | PASS |
| `SRUKF_DEFAULT_PARITY` | PASS |
| `COMMON_GAUSSIAN_POSTERIOR_GATE` | PASS |

Both non-pass classes are confined to `tests/test_r2_measurement_fidelity.py` and both are known
pre-existing:

1. **1 FAILED** — `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change`:
   `lunar_od/dynamics.py` hashes `b369899a45a8…` against the R2-frozen `dde8c188e035…`. This is the
   Phase 16 explicit-SRP integration into `dynamics.py`. The corresponding R2 byte-gate re-freeze
   was **explicitly withheld** from the Phase 17C-RF owner authorization and remains owner business.
   Classification: `KNOWN_PREEXISTING_PROVENANCE`.

2. **10 ERRORS** — all one module-scoped fixture failure,
   `NameError: name 'CLOSURE_CURRENT_TREE_SHA256' is not defined` at
   `examples/r2_measurement_fidelity_validation.py:1845`. The long-standing R2 protected-file typo.
   Classification: `KNOWN_PREEXISTING_PROVENANCE`.

Neither was fixed, per §56's instruction not to silently repair historical Phase16/R2/FA-06 issues.

The classification rests on something stronger than recollection: R1M modified **zero tracked
files**, verified by an empty `git status --porcelain --untracked-files=no` throughout. No non-pass
can be R1M-introduced when no tracked byte changed.

```
NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES = 0
```

## 38. Full-Suite Results

```
TOTAL_TESTS_COLLECTED = 1245
TESTS_PASSED          = 1204
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
```

Every non-pass classified, per §56:

| non-pass | count | classification |
|---|---:|---|
| `test_r2_measurement_fidelity.py` module fixture — `NameError: CLOSURE_CURRENT_TREE_SHA256` | 10 errors | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change` — Phase 16 `dynamics.py` bytes | 1 failed | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_fa06_pytest_session_imports_isolated_repository` — root-`pytest.ini` pythonpath | 1 failed | `KNOWN_PREEXISTING_ENVIRONMENT` |
| slow/optional regressions gated behind `LUNAR_OD_RUN_SLOW_TESTS=1` and optional dependencies | 29 skipped | `KNOWN_PREEXISTING_ENVIRONMENT` |

The FA-06 failure did not appear in the targeted battery because that battery did not include
`test_measurement_model_safety_diagnostics.py`. It was investigated rather than assumed
pre-existing. Its mechanism is the root-`pytest.ini` trap: pytest resolved `rootdir` to the
*workspace* root `C:\Users\erayh\Documents\Python\Grad` and picked up that directory's `pytest.ini`,
which injects the **sibling worktree** `python_port` and `python_port/desktop_app` onto
`pythonpath`. The test's own diagnostic output confirms the important part — its first assertion
passes, and `lunar_od.__file__` resolves to
`C:\Users\erayh\Documents\Python\Grad\python_port_phase17r0\lunar_od\__init__.py`. The correct
package was imported from the correct worktree; only the `getini("pythonpath") == []` assertion
fails. No scientific result in this phase is affected.

The workspace-root `pytest.ini` lies outside the repository and belongs to the multi-worktree
workspace, so it was not touched (workspace `CLAUDE.md`: do not modify a non-target worktree's
state). This matches the non-pass baseline R1G recorded, which listed exactly the Phase16 / R2 /
FA-06 classes.

```
KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1M_INTRODUCED_NONPASSES                = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0
```

## 39. What This Phase Actually Established

1. The estimator-reported `sigma_K` is set by the eigenvalue safety floor, not by data or prior —
   reproduced to `1.5e-10` and proven by a scale-invariance control.
2. All previously reported K uncertainties in this project were therefore floor-dominated and
   optimistic; the true data-only values are 12× to 260× larger.
3. The 6/7 rank deficiency is a normal-matrix artifact; `cond(info) = cond(Hs)^2` exactly, and the
   design matrix is resolvable with seven orders of margin.
4. Pure multi-arc processing is strictly worse for K than the contiguous tracking already in use.
5. The Schur sum rule proves pure multi-arc can only accumulate K information, never rotate K out
   of the local state subspace — verified to `4.3e-07` or better in all three campaigns.
6. Elapsed dynamical span, not observation count, is what buys K information: 10×–6500× at matched
   tracking time.
7. The best configuration found still leaves 45% fractional uncertainty on K, at rank 6/7, with the
   weakest mode `0.9999999985` aligned with K.
8. The beneficial configuration requires no production change; the one requiring production change
   is not beneficial.

## 40. What Remains Unknown

- Whether a square-root covariance path recovers a usable K sigma. Strongly suggested by section 11,
  **not demonstrated** — no such path was built or run.
- Whether spans much longer than 7 orbits, over which the Sun direction rotates substantially, break
  the degeneracy. Section 14's saturation is measured only to 5 orbits contiguous / 7 linked.
- Whether eclipse-rich geometry helps. Section 15's mechanism is well-motivated but eclipse entries
  and Sun-geometry rotation were not instrumented — `NOT_RECORDED`, not estimated.
- Whether nonlinear multi-arc recovery behaves as the linear information analysis predicts. Not run,
  by gate.
- Whether R1G's G0 conditional-information value `1.5212` arose from a different calculation or a
  transcription error. R1M establishes it is not reproducible and is inconsistent with R1G's own
  singular value, but not its origin.
- Everything here is one orbit geometry, one spacecraft configuration, one synthetic truth
  (`SYNTHETIC_CAMPAIGN_TRUTH`, never `SPACECRAFT_TRUTH`).

## 41. Decision Tree From Here

Applying §61 in its stated order:

- *IF multi-arc information becomes practically identifiable* → **not satisfied.** Rank 6/7
  everywhere, best `sigma_K/K_truth = 0.45`, `f_perp` saturating at 0.30.
- *ELSE IF only the continuity-linked model works* → **not satisfied.** Model B is better but does
  not reach practical identifiability either; and it is not an architecture to design, since
  production already supports it.
- *ELSE IF the covariance diagnostic reveals a numerical artifact* → **satisfied, decisively.**
  Two independent numerical artifacts were identified and quantified: the eigenvalue floor
  (§9, §10) and the normal-matrix squaring (§11).
- *ELSE* → the additional-observable path (§66) also applies on the identifiability evidence.

Both the third and fourth branches are triggered. The tree's ordering resolves the precedence, and
the ordering is correct on the merits: the covariance finding is the stronger result (reproduced to
`1.5e-10`, with a bitwise-invariance control) and it is also a **prerequisite**, because any further
K observability experiment would report a formal sigma that is currently an artifact. Fixing the
metric before running the next experiment is the only order that produces interpretable results.

Accordingly the primary next phase is the covariance numerical qualification phase of §65, with the
additional-observable study of §66 as the follow-on once K uncertainties can be trusted.

## 42. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 7acb10ac511ac01ffb0f2aed08034ed403e9bd51
START_TREE   = 68f2a1af1f9f271ade72ab6354071d79e4594bb7

CONTENT_COMMIT = f2af3045e44860b0892fe84b788ab7b75c7c63a2
CONTENT_TREE   = 0fc06af901fbbb7ae0bf4b5dd405eaf5d3213ef1
    (parent = START_HEAD; contains the analysis scripts, artifacts, figures
     and this report)

FINAL_HEAD = the manifest commit, which is the child of CONTENT_COMMIT
FINAL_TREE = the manifest commit's tree
    A commit cannot contain its own hash, so these two values are reported in
    the phase closure message rather than embedded here. Everything else in
    this block is self-contained, and artifacts/r1m_manifest.json records
    CONTENT_COMMIT above.

R1M_INPUT_GATE = PASS

G0_REPRODUCED = YES
G1_REPRODUCED = YES

SIGMA_K_G0_ESTIMATOR       = 3.4688695056e-03
SIGMA_K_G0_DIRECT          = 9.0401754627e-01
SIGMA_K_G0_SVD             = 5.5425845998e-10
SIGMA_K_G0_SCHUR           = 9.0401755679e-01
SIGMA_K_G0_FULL_POSTERIOR  = 8.9531531979e-01

SIGMA_K_G1_ESTIMATOR       = 1.6448102377e-03
SIGMA_K_G1_DIRECT          = 1.9394564364e-02
SIGMA_K_G1_SVD             = 3.7495863432e-10
SIGMA_K_G1_SCHUR           = 1.9394564345e-02
SIGMA_K_G1_FULL_POSTERIOR  = 1.9390722040e-02

G0_FLOORED_MODES = 1 of 7
G1_FLOORED_MODES = 1 of 7

G0_FLOOR_CONTROLS_K_VARIANCE = YES   (predicted vs reported, rel 1.5e-10)
G1_FLOOR_CONTROLS_K_VARIANCE = YES   (predicted vs reported, rel 6.2e-09)

SCALING_PHYSICAL_INVARIANCE_GATE = PASS

RANK_TOLERANCE_CLASS = NUMERICALLY_NEAR_SINGULAR
    (normal-matrix domain; WEAK_BUT_NONZERO in the square-root domain, see section 11)

PRIOR_DIAGNOSTIC_STATUS        = RESOLVED_NUMERICAL_FLOOR_EFFECT
ESTIMATOR_SIGMA_K_TRUST_STATUS = NOT_RELIABLE_IN_NEAR_SINGULAR_REGIME
COVARIANCE_DIAGNOSTIC_GATE     = PASS

G0_K_ORTHOGONAL_FRACTION = 0.008885
G1_K_ORTHOGONAL_FRACTION = 0.224629
G2_K_ORTHOGONAL_FRACTION = 0.243436
G3_K_ORTHOGONAL_FRACTION = 0.269044

SINGLE_ARC_IDENTIFIABILITY_CLASS = WEAK_BUT_NONZERO_AND_SATURATING

MULTI_ARC_CASES = M1, M2, M3 (pre-declared) + F2, F3 (fairness control)

M1_ARCS                        = 2
M1_ELAPSED_SPAN                = 2.50 orbits
M1_RANK                        = Model A 12/13, Model B 6/7
M1_WEAKEST_MODE_K_COMPONENT    = Model A 0.9999999998, Model B 0.9999999923
M1_CONDITIONAL_K_INFORMATION   = Model A 2.628563e-03, Model B 3.478595e+03
M1_K_ORTHOGONAL_FRACTION       = Model A 0.007983, Model B 0.199315

M2_ARCS                        = 3
M2_ELAPSED_SPAN                = 5.50 orbits
M2_RANK                        = Model A 18/19, Model B 6/7
M2_WEAKEST_MODE_K_COMPONENT    = Model A 0.9999999997, Model B 0.9999999970
M2_CONDITIONAL_K_INFORMATION   = Model A 4.006460e-03, Model B 3.658760e+04
M2_K_ORTHOGONAL_FRACTION       = Model A 0.008038, Model B 0.300283

M3_ARCS                        = 5
M3_ELAPSED_SPAN                = 5.50 orbits
M3_RANK                        = Model A 30/31, Model B 6/7
M3_WEAKEST_MODE_K_COMPONENT    = Model A 0.9999999995, Model B 0.9999999985
M3_CONDITIONAL_K_INFORMATION   = Model A 6.477955e-03, Model B 4.902650e+04
M3_K_ORTHOGONAL_FRACTION       = Model A 0.007974, Model B 0.289965

MULTI_ARC_SCHUR_SUM_RULE = PASS   (rel 4.31e-07, 7.82e-08, 4.12e-07)

MODEL_A_INDEPENDENT_ARCS_CLASS  = STRICTLY_WORSE_THAN_EXISTING_CONTIGUOUS_ARC
MODEL_B_CONTINUITY_LINKED_CLASS = BEST_TESTED_BUT_STILL_NOT_PRACTICALLY_IDENTIFIABLE

MULTI_ARC_INFORMATION_CLASS = INFORMATION_ACCUMULATION_ONLY

MULTI_ARC_PRODUCTION_IMPLEMENTATION_JUSTIFIED = NO

MULTI_ARC_LINEAR_GAUSSIAN_ORACLE = NOT_RUN_BY_GATE
MULTI_ARC_NOISE_FREE_RECOVERY    = NOT_RUN_BY_GATE
MULTI_ARC_HOLDOUT_GATE           = NOT_RUN_BY_GATE

OPERATIONAL_JUSTIFICATION_CLASS = NOT_JUSTIFIED_FOR_PRODUCTION_INTEGRATION
RESEARCH_VALUE_CLASS            = APPLICATION_SPECIFIC_RESEARCH_RESULT

ACADEMIC_LITERATURE_ALIGNMENT = CONSISTENT_WITH_ESTABLISHED_OD_THEORY
POTENTIAL_NOVELTY_CLASS       = APPLICATION_SPECIFIC_METHODOLOGICAL_CONTRIBUTION

PRODUCTION_CODE_CHANGED        = NO
PHYSICAL_MODEL_CHANGED         = NO
MEASUREMENT_MODEL_CHANGED      = NO
ESTIMATOR_ARCHITECTURE_CHANGED = NO

P21_REGRESSION                           = PASS
MODEL_S_REGRESSION                       = PASS
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

TOTAL_TESTS_COLLECTED = 1245
TESTS_PASSED          = 1204
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29

KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1M_INTRODUCED_NONPASSES                = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0

MAIN_CHANGED        = NO
ORIGIN_MAIN_CHANGED = NO

REPORT_COMPLETENESS_GATE = PASS
    all 43 mandatory sections present.
    §57 twelve-part format in full: sections 7, 8, 13.
    §57 abbreviated (purpose / method / results / interpretation / verdict):
      sections 9, 10, 11, 17, 18, 19, 20, 21 -- the shared setup, "what
      changed" and system-impact answers are identical to section 7's and are
      not restated per test.
    §53 artifact set complete except r1m_holdout.csv, correctly withheld by
      the information gate.
    §54: all 7 required figures produced, plus 3 supporting.

PHASE17_R1M_GATE = CHARACTERIZATION_COMPLETE

PRIMARY_CLASS =
  K_SRP_FORMAL_UNCERTAINTY_IS_A_NUMERICAL_FLOOR_ARTIFACT
  AND_MULTI_ARC_ACCUMULATES_BUT_DOES_NOT_SEPARATE_K

NEXT_ACTION =
  COVARIANCE_NUMERICAL_QUALIFICATION_PHASE (§65), then
  PHASE_17_R1O_ADDITIONAL_OBSERVABLE_FEASIBILITY (§66).
  Neither begins automatically; both require separate authorization.

COMMITS_CREATED = 2
COMMIT_LIST =
  f2af3045e44860b0892fe84b788ab7b75c7c63a2  analysis scripts, artifacts,
                                            figures and this report
  <manifest commit>                         artifacts/r1m_manifest.json and
                                            this reference

PUSH = NONE
MERGE = NONE
FORCE_PUSH = NONE
MAIN_MODIFICATION = NONE
HISTORY_REWRITE = NONE
```

## 43. Exact Next Action

**STOP.** No production multi-arc estimator, no new measurement types, no merge, no push, no change
to main (§67).

The next authorized phase should be the **covariance numerical qualification phase** of §65 —
qualifying the covariance infrastructure before any formal K sigma is used again. It has an unusually
concrete starting point from section 11: the codebase already contains a qualified square-root
information filter, and `cond(info) = cond(Hs)^2` was measured exactly, so the remedy to evaluate is
computing K covariance from the SRIF's `R` factor rather than from an eigen-decomposition of
`H^T W H`.

The additional-observable feasibility study (§66, Phase 17-R1O) follows once K uncertainties can be
trusted. It must not start first: it would report the same artifact.

Neither should begin automatically. Both require separate authorization.

### The phase's question, answered directly

> Is multi-arc processing mathematically capable of making `K_SRP` practically identifiable using
> the measurement physics we already trust?

**No.** Pure multi-arc is provably incapable of it — the Schur sum rule shows it can only add up
K information that already exists arc by arc, never create an independent K direction — and it is
strictly worse than the contiguous tracking already in production. The continuity-linked
alternative is better and needs no new code, but still leaves 45% fractional uncertainty at rank
6/7.

Academically this confirms established multi-arc theory in a new application: the method works for
parameters with arc-crossing secular signatures, and `K_SRP` over a few lunar orbits is not one.
Operationally it means the multi-arc estimator should not be built for this purpose, elapsed span is
a better lever than tracking volume, and — most consequentially — the formal K uncertainties this
project has been reporting are numerical artifacts that must be qualified before they are used
again.
