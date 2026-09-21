# PHASE 17-R1O-R — SCIENTIFIC ERRATUM, STM LAYOUT REPAIR, AND HISTORICAL R1O REQUALIFICATION

## 1. Executive Summary

Phase 17-R1O-OPT found a defect in the already-closed Phase 17-R1O analysis code. This phase
repaired it, proved the repair against an independent oracle, reproduced R1O's published numbers
exactly, regenerated them correctly, and determined what R1O should have concluded.

One line was wrong. `examples/phase17_r1o_core.py::chain_to_augmented_columns` unflattened the
column-major 6x6 state-transition matrix with NumPy's default C order, which for a column-major
flattening returns `Phi^T`. Phi is not symmetric, so every surrogate state-design row R1O published
was built from the transpose of the state-transition matrix. The K columns were never affected:
`S_K` is a plain 6-vector that is never reshaped.

**The historical result was reproduced exactly before anything was corrected.** Re-running R1O's own
published configuration through the defective path gives `f_perp = 0.875701` against a published
`0.8757`, `0.441901` against a published `0.4419`, and `I(K|x) = 2.658848e+09` and `6.675259e+07`
against the same digits in R1O's verdict block — bitwise agreement with the literal pre-repair
expression. This requalification is therefore genuinely re-running the path that produced the
published result.

**Corrected, R1O's headline collapses.** The landmark surrogate at 1 urad falls from `f_perp = 0.8757`
to **`0.2530`**, against a range-only baseline of `0.2523` — a direction gain of `+0.0007`, not a
rotation from 15 deg to 61 deg. The DDOR surrogate at 1 nrad falls from `0.4419` to **`0.0701`**,
*below* the baseline. The Earth-LOS negative control stays negative (`0.2617 -> 0.2578`).

**The `sigma_K` improvements are real; they were never the thing that was wrong.** Corrected landmark
observations still drive fractional K uncertainty from 9.27% to 0.67%. What the defect did was make
an information-*magnitude* result look like an information-*direction* result. R1O's own framework
already distinguished these; the corrupted state basis is what blurred them.

**A downstream conclusion is revised, though not reversed.** Phase 17-R1O-D concluded that production
DDOR disagreed with R1O's surrogate because the surrogate discarded differential light-time. Its
production result stands — it was re-run here in full and reproduces every published value — but with
Phi corrected, R1O's surrogate lands almost exactly on production (1 nrad: `0.0701` vs `0.0738`;
5 nrad: `0.2070` vs `0.2121`). The STM defect accounts for **97-99%** of the gap R1O-D attributed to
event physics. R1O-D could not have seen this: its bridge experiment varied event physics between two
arms that both used the defective Phi.

`PHASE17_R1O_R_GATE = PASS`.
`PRIMARY_CLASS = HISTORICAL_ANALYSIS_DEFECT_REPAIRED_AND_R1O_SCIENTIFIC_CONCLUSIONS_REQUALIFIED`.

---

## 2. Why This Repair Phase Exists

Phase 17-R1O closed as `PASS` with `PROMISING_OBSERVABLE_FOUND = YES` and two Tier-1 candidates
ranked for follow-up. Phase 17-R1O-D took the first candidate to production and found it did not
reproduce. Phase 17-R1O-OPT took the second to production, and while qualifying its analytic
Jacobian against re-propagated finite differences found state columns disagreeing by factors of
`1e2`-`1e8` — and traced the cause not to its own new code but to R1O's committed analysis helper.

R1O-OPT deliberately did not repair it: repairing a closed phase's published science is an
owner-authorized act, not a side effect of a different phase. This phase is that authorization
exercised.

This phase does not ask which observable makes K_SRP easier to estimate. It asks only:

> After repairing the known STM unpacking defect, what should Phase 17-R1O actually have concluded?

## 3. What Phi / STM Means

The state-transition matrix `Phi(t, t0)` maps a small initial-state perturbation forward:

```
dx(t) = Phi(t, t0) dx0
```

For the six-state orbital vector `x = [x, y, z, vx, vy, vz]^T`, Phi is 6x6, and the measurement
Jacobian with respect to the arc-initial state is

```
H_x0 = H_x(t) Phi(t, t0)
```

R1O's observables are functions of spacecraft inertial position at a single time, `g = g(r_sc(t))`,
so their chain is

```
dg/dx0 = dg/dr(t) . Phi[:3, :](t)
dg/dK  = dg/dr(t) . S_K[:3](t)
```

Note the asymmetry that determines this entire phase: the state chain goes through a **matrix** that
must be unflattened; the K chain goes through a **vector** that must not.

## 4. Why Storage Order Matters

`lunar_od.dynamics` packs the STM column-major. For a column-major flattening `v` of Phi:

```
v[i + 6j] = Phi[i, j]           (pack,   order="F")
M[a, b]   = v[6a + b]           (unpack, order="C")
          = Phi[b, a]      ==>  M = Phi^T
```

Reading a column-major buffer in row-major order is a transpose. For a symmetric matrix this would be
invisible; Phi is not symmetric, and s8's oracle checks that explicitly before trusting anything else.

Substituting `Phi^T` does not perturb `f_perp` slightly. `f_perp` measures how much of the K design
column lies **outside the span of the six state columns**, so replacing the state columns replaces the
entire subspace K is measured against. A wrong state basis can manufacture an arbitrary appearance of
parameter observability while the parameter sensitivity itself stays numerically perfect.

## 5. Entering Repository State

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 887efa62b0dd44e916036313bc39ea6e49422524
START_TREE   = 4df2e2d22130edc87aa6e8169172bdf7b950a33e

main         = fea476f81dad07b3914e53eab709e10fa6e9d10b
origin/main  = fea476f81dad07b3914e53eab709e10fa6e9d10b
```

Canonical main matched the required hash on both local and remote. Working tree carried only the two
pre-existing untracked R1O-D orphan scripts (`phase17_r1od_fd_convergence.py`,
`phase17_r1od_qualification.py`), which R1O-D deliberately left uncommitted because they reference a
superseded API; this phase left them untouched.

`R1O_REPAIR_INPUT_GATE = PASS`

## 6. STM Storage Contract

Established from production sources before editing any analysis code.

| location | operation | order |
|---|---|---|
| `dynamics.py` s1249 | pack `eye(6)` initial condition | `order="F"` |
| `dynamics.py` s1255 | unpack in the 48-state RHS | `order="F"` |
| `dynamics.py` s1292 | pack `phi_dot` in the 48-state RHS | `order="F"` |
| `dynamics.py` s394/s421 | 42-state RHS unpack/pack | `order="F"` |
| `dynamics.py` s386 | docstring | *"column-major order, matching MATLAB's `Phi(:)`"* |
| `two_way_range.py` s512, `delta_dor.py` s411, `radiometrics.py` s944, `measurements.py` s1101, `estimators.py` s1903, `observability.py` s377/s395, `accelerated.py` s448/s464, `scenarios.py` s1501/s1547, `two_way_counted_doppler_reference.py` s522 | consumer unpack | `order="F"` |

Every production producer and every production consumer agrees. The convention is not one
implementation's habit; it is uniform across the codebase and documented in the propagator itself.

```
STM_PACK_ORDER   = column-major (order="F")
STM_UNPACK_ORDER = column-major (order="F")
STM_LAYOUT_CONTRACT_GATE = PASS
```

One apparent exception was checked rather than assumed:
`lunar_od/lunar_landmark_optical.py` s270 calls `.reshape(6, 6)` without an order. It takes an
**already-2-D** Phi as an argument, where the call is a no-op; all its callers pass
`.reshape((6, 6), order="F")` explicitly, and `tests/test_lunar_landmark_optical.py` already contains
a transpose-sensitivity test. Not a defect.

## 7. Defect Reproduction

`examples/phase17_r1o_core.py::chain_to_augmented_columns`, as committed through R1O, R1O-D and
R1O-OPT:

```python
phi = np.asarray(nom48_row[6:42], dtype=float).reshape(6, 6)    # C order
s_k = np.asarray(nom48_row[42:48], dtype=float)
h_x0 = dg_dr @ phi[:3, :]
h_k  = dg_dr @ s_k[:3]
```

The docstring asserted the wrong convention as well — *"[6:42] Phi (6x6, row-major)"* — which is
presumably how the error survived review. Both the code and the docstring were repaired.

## 8. Independent STM Oracle

`examples/phase17_r1o_r_stm_oracle.py`. Phi is *defined* by `dx(t) = Phi dx0`, so its position rows
satisfy `dr(t)/dx0[:, j] = Phi[:3, j]`. The oracle measures the left-hand side directly, by central
differences on trajectories re-propagated with `propagate_state` — the **6-state** propagator, which
never forms a variational block at all — and compares both candidate unpackings against it, at two
real campaign epochs on the frozen campaign trajectory.

The comparison is **floor-limited**, so it is judged against the finite difference's own measured
precision (the Phase 17A-R floor-consistency methodology) rather than a fixed absolute threshold.
Asking a candidate to be more accurate than the oracle judging it is not a test.

| epoch | Phi symmetric? | FD floor (step-halving) | `order="F"` rel err | `order="C"` rel err | F / floor | C / F |
|---|---|---:|---:|---:|---:|---:|
| mid-arc (t = 3540 s) | **No** | 1.311e-05 | 8.878e-06 | 1.000000 | **0.677** | 1.13e+05 |
| end-arc (t = 7080 s) | **No** | 3.580e-05 | 2.590e-05 | 1.000000 | **0.724** | 3.86e+04 |

Per-component relative error against FD truth, mid-arc:

| component | `order="F"` | `order="C"` |
|---|---:|---:|
| x | 5.19e-07 | 9.69e-01 |
| y | 1.85e-04 | 2.17e+00 |
| z | 7.01e-06 | 5.37e+00 |
| vx | 1.56e-05 | 1.00e+00 |
| vy | 4.95e-07 | 1.00e+00 |
| vz | 3.22e-06 | 1.00e+00 |

`order="F"` sits *below* the oracle's own resolution — it is indistinguishable from truth at the
precision available. `order="C"` is wrong by 100% and separated from the F-order error by four to
five orders of magnitude. The storage contract is established by measurement, not by reading code.

`R1O_STM_LAYOUT_ORACLE_GATE = PASS`

## 9. Minimal Code Repair

One line and its docstring. No other R1O code was touched, so every delta reported later is
causally attributable to the storage order alone.

```python
-    phi = np.asarray(nom48_row[6:42], dtype=float).reshape(6, 6)
+    phi = np.asarray(nom48_row[6:42], dtype=float).reshape((6, 6), order="F")
```

Repair commit `75ba49a7e86cf7c4ea01e45ca757126159dc0654`.

## 10. Permanent Regression Test

`tests/test_r1o_stm_layout.py`, five behavioural tests — not source-text matching, which would pass
against any rewrite that reintroduced the same arithmetic differently.

1. **The propagated Phi is asserted non-symmetric first.** A symmetric Phi would make every other
   assertion in the file vacuous, and Phi *is* symmetric at `t = 0`, where it equals the identity.
   This is the reason the defect survived: the obvious test is the blind one.
2. Each of the six columns of the `order="F"` unpacking is checked against re-propagated central
   differences, and the C-order unpacking is *required to be decisively wrong* — so the test fails if
   it ever stops being able to tell the two apart.
3. The helper must reproduce `dg_dr @ Phi[:3, :]` and is rejected if it returns the transpose.
4. End-to-end: the helper's state columns against re-propagated trajectories — the assertion that
   would have caught the published defect.
5. The K column is shown storage-order-independent by construction: transposing the Phi block leaves
   `h_k` bitwise unchanged.

Hermetic — no SPICE, no campaign data — so it runs in the ordinary suite. **Verified by mutation:**
reintroducing `reshape(6, 6)` fails tests 3 and 4 (`helper column 0 rel error 6.220e-02`); restoring
the repair returns all five to green.

`R1O_STM_LAYOUT_REGRESSION_GATE = PASS`

## 11. Blast Radius

Determined by exhaustive search of every `reshape(6,6)`-shaped expression in the repository, then
verified per consumer — not inherited from R1O-OPT's list.

```
AFFECTED_ANALYSES =
  phase17_r1o_core.build_landmark_arc        (R1O landmark surrogate)
  phase17_r1o_core.build_ddor_arc            (R1O DDOR-like surrogate)
  phase17_r1o_celestial                      (Earth-center LOS negative control)
  phase17_r1od_surrogate_bridge              (surrogate side ONLY)

UNAFFECTED_ANALYSES =
  R1O range-only baseline  (production two_way_range_nominal_and_initial_jacobian
                            + _two_way_range_k_srp_column, both order="F")
  production DDOR          (lunar_od/delta_dor.py -- re-run in full, s18)
  production optical       (lunar_od/lunar_landmark_optical.py -- callers pass
                            order="F"; transpose-sensitivity test already present)
  ALL K columns everywhere (S_K is a 6-vector, never reshaped -- s12)
  Phase 17-R1M, Phase 17-R1COV  (no use of the helper)
```

After the repair, the only remaining C-order `reshape(6, 6)` occurrences in the repository are
(a) the already-2-D argument in `lunar_landmark_optical.py` and (b) deliberate negative controls in
this phase's own oracle, requalification script and regression test. No second defect exists.

## 12. K-Column Invariance

Structural, then measured. `chain_to_augmented_columns` reads `s_k = nom48_row[42:48]` — a plain
6-vector slice. No 2-D reshape occurs, so no storage-order convention can apply to it.

Measured on the 6568-row landmark design, the 143-row DDOR design, the 590-row Earth-LOS design and
the 143-row bridge design: the K column is **bitwise identical** before and after the repair in every
case (`max |diff| = 0.0`), while the state columns change.

```
K_COLUMN_INVARIANCE_GATE = PASS
```

This is why R1O's `sigma_K` values were *directionally* misleading but never *arithmetically*
fabricated, and why the defect was invisible to the K-sensitivity qualification work of Phase 17A-R
and Phase 17-R.

## 13. Historical Landmark Reproduction

The repaired helper is fed a `nom48` whose `[6:42]` block has been replaced by `vec_F(Phi^T)`. This
reproduces the defective arithmetic exactly without un-repairing the shared code, and it is verified
against the *literal* pre-repair expression rather than assumed:

```
max |state column diff| = 0.000e+00
max |K column diff|     = 0.000e+00      bitwise = True
```

On R1O's own published configuration — 15 orbits, all three DSN stations, 90 s cadence, **782 range
observations**, 1179 trajectory samples:

| quantity | R1O published | reproduced here |
|---|---:|---:|
| range-only `f_perp` | 0.2523 | **0.252292** |
| range-only `sigma_K/K` | 9.27% | **9.2735%** |
| range-only `I(K\|x)` | 1,162,817.09 | **1.162817e+06** |
| landmark 1 urad `f_perp` | 0.8757 | **0.875701** |
| landmark 1 urad `sigma_K/K` | 0.1939% | **0.193934%** |
| landmark 1 urad `I(K\|x)` | 2.658848e+09 | **2.658848e+09** |
| DDOR 1 nrad `f_perp` | 0.4419 | **0.441901** |
| DDOR 1 nrad `sigma_K/K` | 1.224% | **1.223956%** |
| DDOR 1 nrad `I(K\|x)` | 6.675259e+07 | **6.675259e+07** |

Every published digit reproduces.

`R1O_HISTORICAL_LANDMARK_REPRODUCTION_GATE = PASS`

## 14. Corrected Landmark Result

Identical arc, landmark set, epochs, noise, K truth and sampling. Only Phi's storage order differs.
6568 landmark design rows in both runs.

| | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` |
|---|---:|---:|---:|---:|
| range only | 1.162817e+06 | 0.252292 | 14.6132 deg | 9.2735% |
| landmark 1 urad — historical (`Phi^T`) | 2.658848e+09 | 0.875701 | 61.1280 deg | 0.1939% |
| landmark 1 urad — corrected (`Phi`) | 2.218707e+08 | **0.252964** | 14.6530 deg | 0.6714% |

The direction gain over the range-only baseline is `+0.000672` in `f_perp`, or `+0.04 deg` in
`theta_K`. The published claim was a rise to `0.8757` / `61.13 deg`.

Conditional K information still rises by 191x, and `sigma_K/K` still falls from 9.27% to 0.67%. The
observable genuinely helps — it simply helps by adding magnitude along a direction the six-state
subspace already largely spans, not by supplying a new direction.

## 15. Landmark Attribution

```
delta f_perp from the STM repair alone            = -0.622737
```

Phase 17-R1O-OPT independently measured `+0.623473` for the same transposition effect and `-0.006998`
for the production-measurement-model difference, on a single-station range baseline. This phase
reproduces the transposition term to `7.4e-04` on R1O's own three-station baseline, from an
independent script. The two phases agree that the transposition, not the measurement model, is the
entire story.

The residual attributable to the production measurement model is smaller than the transposition term
by roughly **two orders of magnitude**.

## 16. Historical DDOR Reproduction

Same arc and station pair (Goldstone-Canberra), 143 dual-visible epochs at the qualified 10 deg
elevation mask — the same 143 epochs production used.

| noise | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` |
|---:|---:|---:|---:|---:|
| 1 nrad | 6.675259e+07 | 0.441901 | 26.2252 deg | 1.2240% |
| 5 nrad | 4.119144e+06 | 0.363285 | 21.3021 deg | 4.9272% |

The 1 nrad row reproduces R1O's `DDOR_BEST_F_PERP = 0.4419`; the 5 nrad row reproduces the `0.3633`
in R1O-D's own bridge table. Both historical reproductions are exact.

## 17. Corrected DDOR Result

| noise | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` | delta from repair |
|---:|---:|---:|---:|---:|---:|
| 1 nrad | 1.681884e+06 | **0.070144** | 4.0222 deg | 7.7108% | -0.371757 |
| 5 nrad | 1.337112e+06 | **0.206980** | 11.9454 deg | 8.6480% | -0.156305 |

Both corrected values are **below** the range-only baseline's `0.252292`.

This is not a contradiction. `f_perp = sqrt(I_K|x / I_KK)` is a *fraction*. Adding a heavily weighted
observable whose K signature lies largely inside the state subspace raises the denominator `I_KK`
faster than the numerator `I_K|x`, so the fraction falls even though total information rises. At
1 nrad the DDOR rows are weighted `~25x` more strongly than at 5 nrad, which is exactly why the
1 nrad case is the *worse* of the two on direction while being the better on `sigma_K`.

## 18. DDOR Production Comparison

Production DDOR was re-run in full (`examples/phase17_r1od_campaign.py`, unmodified) to confirm this
phase did not perturb it. It reproduces every published value exactly: range-only `0.2523` /
`1.162817e+06`, 5 nrad `0.2121` / `1.329505e+06`, and the complete noise sweep
(1 nrad `0.0738`, 2 `0.1264`, 3 `0.1657`, 5 `0.2121`, 10 `0.2493`, 30 `0.2636`, 100 `0.2641`).

```
R1OD_PRODUCTION_DDOR_UNCHANGED = YES   (verified by re-run, not asserted)
```

Now the comparison R1O-D could not make:

| noise | surrogate as published | **surrogate, Phi corrected** | production | STM term | event-physics term | STM share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 nrad | 0.441901 | **0.070144** | 0.0738 | -0.371757 | +0.003656 | **99.0%** |
| 5 nrad | 0.363285 | **0.206980** | 0.2121 | -0.156305 | +0.005120 | **96.8%** |

Per s18's explicit instruction, the two effects are separated rather than collapsed:

- **The event-physics difference is real and is not zero.** Production's common-transmit-event
  treatment does move the answer relative to the corrected simultaneous-epoch surrogate.
- **But it is small, and it acts in the opposite direction.** Production is marginally *better* than
  the corrected surrogate (`+0.0037` and `+0.0051`), not worse. R1O-D's picture — a surrogate
  flattered by discarded light-time — is inverted: with Phi corrected, discarding the differential
  light-time makes the surrogate slightly *pessimistic*, not optimistic.
- **The STM defect dominates by a factor of ~30-100.**

R1O-D's *conclusion* is untouched: production DDOR is `MAGNITUDE_ONLY` and sits below the range-only
baseline at every tested precision. Only its *causal explanation* of the surrogate-production gap is
superseded.

The methodological reason R1O-D reached the wrong cause is worth stating plainly: its bridge
experiment compared R1O's surrogate against production while varying event physics — but **both arms
of the comparison used the defective Phi**. A controlled experiment cannot detect a defect its
control and treatment share. It correctly identified a real difference between the two models and
then attributed to that difference an effect that was overwhelmingly caused by something common to
both.

## 19. Earth-LOS Historical vs Corrected

| | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` |
|---|---:|---:|---:|---:|
| historical | 1.250870e+06 | 0.261667 | 15.1690 deg | 8.9412% |
| corrected | 1.214443e+06 | 0.257829 | 14.9413 deg | 9.0743% |

```
NUMERIC CHANGE              = YES, small (-0.003838 in f_perp)
SCIENTIFIC CLASSIFICATION   = UNCHANGED (CHARACTERIZED_NEGLIGIBLE_GAIN)
```

The negative control remains negative. Its gain over the `0.252292` baseline falls from `+0.0094` to
`+0.0055` — still negligible, and still the correct scientific reading. This is a meaningful check:
had the control flipped sign or become large, it would have indicated the repair broke something
rather than fixed it.

## 20. R1O-D Bridge Historical vs Corrected

The surrogate side only. Production DDOR is not touched.

| | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` |
|---|---:|---:|---:|---:|
| bridge reduced-mode — historical | 1.242473e+06 | 0.260790 | 15.1169 deg | 8.9713% |
| bridge reduced-mode — corrected | 1.163837e+06 | 0.252403 | 14.6197 deg | 9.2694% |

(The bridge's reduced-mode weighting uses an approximate constant perpendicular baseline for a
side-by-side, as its own docstring notes; the authoritative surrogate numbers are s16/s17's, which
use R1O's exact per-row `B_perp` weighting.)

The bridge interpretation changes as described in s18: its measured *difference* between the two
models was real, but its *attribution* of that difference to event physics was wrong by a factor of
~30-100.

## 21. Corrected Cross-Observable Table

All rows on one arc: 15 orbits, three DSN stations, 90 s cadence, 782 range observations.

| case | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` | state-Jacobian source | classification |
|---|---:|---:|---:|---:|---|---|
| range only | 1.162817e+06 | 0.252292 | 14.6132 | 9.2735% | production (`order="F"`) | baseline |
| DDOR surrogate 1 nrad | 6.675259e+07 | 0.441901 | 26.2252 | 1.2240% | R1O surrogate (`Phi^T`) | **superseded** |
| DDOR surrogate 1 nrad | 1.681884e+06 | 0.070144 | 4.0222 | 7.7108% | R1O surrogate (`order="F"`) | corrected |
| DDOR surrogate 5 nrad | 4.119144e+06 | 0.363285 | 21.3021 | 4.9272% | R1O surrogate (`Phi^T`) | **superseded** |
| DDOR surrogate 5 nrad | 1.337112e+06 | 0.206980 | 11.9454 | 8.6480% | R1O surrogate (`order="F"`) | corrected |
| landmark surrogate 1 urad | 2.658848e+09 | 0.875701 | 61.1280 | 0.1939% | R1O surrogate (`Phi^T`) | **superseded** |
| landmark surrogate 1 urad | 2.218707e+08 | 0.252964 | 14.6530 | 0.6714% | R1O surrogate (`order="F"`) | corrected |
| Earth LOS 1 urad | 1.250870e+06 | 0.261667 | 15.1690 | 8.9412% | R1O surrogate (`Phi^T`) | **superseded** |
| Earth LOS 1 urad | 1.214443e+06 | 0.257829 | 14.9413 | 9.0743% | R1O surrogate (`order="F"`) | corrected |
| **production DDOR 1 nrad** | — | 0.0738 | — | 7.82% | production (`order="F"`) | reference (R1O-D, re-run) |
| **production DDOR 5 nrad** | 1.329505e+06 | 0.2121 | 12.25 | 8.67% | production (`order="F"`) | reference (R1O-D, re-run) |

Production optical (R1O-OPT) is reported on its own single-station range baseline
(`f_perp` 0.29499 -> 0.325272, `sigma_K/K` 3.4954% -> 2.9746%, `MARGINAL_DIRECTION_GAIN`) and is
**not** placed in this table as a directly comparable row, because its baseline differs. Its
qualitative agreement with the corrected landmark surrogate is the meaningful comparison, and it
holds: both find a marginal direction gain where R1O reported a decisive one.

## 22. Explaining `f_perp`

`f_perp` is not a quality score. It answers one specific question:

> Of the signature that a change in K_SRP leaves in the measurements, what fraction **cannot be
> mimicked** by adjusting the six initial position and velocity components instead?

```
f_perp = ||b_K,perp|| / ||b_K|| = sqrt( I_K|x / I_KK )
```

- **Small `f_perp`** — K's effect on the measurements looks much like a state correction. An
  estimator can absorb a K error into the orbit solution, so K is weakly identifiable no matter how
  much data is collected.
- **Large `f_perp`** — K produces a measurement pattern the orbit cannot imitate, so the data can
  distinguish the two.

Two consequences visible in this phase's tables:

1. `f_perp` can **fall** while total information **rises** (s17's DDOR rows). More data along a
   direction the state already spans inflates `I_KK` faster than `I_K|x`.
2. `sigma_K` can **improve substantially** while `f_perp` is flat (s14's landmark rows: 9.27% ->
   0.67% at `f_perp` `0.2523` -> `0.2530`). That is a magnitude result, and it is a real operational
   gain — it is simply not the *direction* result R1O claimed.

## 23. What R1O Originally Claimed

```
PROMISING_OBSERVABLE_FOUND         = YES
BEST_INFORMATION_DIRECTION_CLASS   = STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION (both)
BEST_OBSERVABLE_F_PERP             = 0.8757   (landmark LOS, 1 urad)
DDOR_BEST_F_PERP                   = 0.4419   (1 nrad)
SELECTED_CANDIDATE_1               = DDOR-like plane-of-sky angle
SELECTED_CANDIDATE_2               = Lunar landmark LOS
```

with the landmark case described as *"the single strongest result in the entire K_SRP investigation
to date"* and the `15 deg -> 60 deg` rotation as *"substantial ... not marginal."*

## 24. What R1O Should Have Claimed

> On this trajectory, neither a DDOR-like plane-of-sky angle nor a lunar-landmark line of sight
> meaningfully rotates K_SRP's information direction out of the six-state subspace, at any tested
> precision. The landmark case adds a large amount of K information *magnitude* — enough to take
> `sigma_K/K` from 9.27% to 0.67% at 1 urad — but leaves `f_perp` at the baseline's own `~0.25`.
> The DDOR case reduces `f_perp` below the baseline. The Earth-center LOS control, as originally
> reported, adds negligible information.

The published ranking (DDOR first, landmark second) rested entirely on transposed state Jacobians and
does not survive. Neither candidate beats the baseline it was measured against on direction.

## 25. Corrected Scientific Classification

```
R1O_ORIGINAL_CLASS  = STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION (both candidates)
R1O_CORRECTED_CLASS = NO_STRONG_COMPLEMENTARY_OBSERVABLE_FOUND
                      + MARGINAL_INFORMATION_DIRECTION_CHANGES_ONLY
                      + LANDMARK_SURROGATE_AGREES_WITH_PRODUCTION_AFTER_STM_REPAIR
                      + DDOR_SURROGATE_STILL_DIFFERS_FROM_PRODUCTION_DUE_TO_EVENT_PHYSICS
                        (real, but ~30-100x smaller than the STM term)

R1O_ORIGINAL_RANKING_VALID = NO
```

The original ranking is not preserved for continuity. It is withdrawn.

## 26. Affected Downstream Statements

- **R1O's candidate selection and ranking** — withdrawn.
- **R1O-D s21's causal attribution** — superseded (s18). Its production result is not.
- Any statement anywhere that a lunar-landmark or DDOR observable *rotates K's direction* on this
  trajectory.

Provenance notices were added to the headers of `phase17_r1o_additional_observable_feasibility.md`
(full supersede notice), `phase17_r1od_production_ddor_qualification.md` (result valid, attribution
revised) and `phase17_r1o_opt_production_optical_qualification.md` (confirmed). No production result
in any downstream report was rewritten.

## 27. Unaffected Downstream Results

Verified, not assumed:

- **R1O's range-only baseline** — `f_perp = 0.252292`, `sigma_K/K = 9.2735%`, reproduced exactly.
- **Production DDOR (R1O-D)** — re-run in full; every published value reproduces.
- **Production optical (R1O-OPT)** — never used the helper; `MARGINAL_DIRECTION_GAIN` confirmed, and
  now independently corroborated from the surrogate side.
- **All K columns, everywhere** — bitwise invariant under the repair.
- **Phase 17-R1M's information-geometry conclusions** — no use of the helper.
- **Phase 17-R1COV's square-root covariance qualification** — no use of the helper; it remains the
  covariance path every `sigma_K` in this phase was computed through.
- **Phase 17-R1O-D's coherent-bias finding** (1 ns unmodelled bias -> 22.5% of K truth) and
  **R1O-OPT's** (200 m coherent catalog bias -> 14.8%) — both computed from K columns and production
  models, untouched.
- **K sensitivity is nonzero** — Phase 17A-R's derivative chain, untouched.

## 28. What Changed in the Repository

```
PRODUCTION_CODE_CHANGED        = NO
ANALYSIS_CODE_CHANGED          = YES  (one line + docstring in examples/phase17_r1o_core.py)
HISTORICAL_REPORT_CHANGED      = YES  (supersede/provenance notices only; no number deleted)

PHYSICAL_MODEL_CHANGED         = NO
MEASUREMENT_MODEL_CHANGED      = NO
ESTIMATOR_ARCHITECTURE_CHANGED = NO
K_SOLVE_FOR_MATH_CHANGED       = NO
COVARIANCE_METHOD_CHANGED      = NO
```

Added: `examples/phase17_r1o_r_stm_oracle.py`, `examples/phase17_r1o_r_requalification.py`,
`tests/test_r1o_stm_layout.py`, `docs/phase17_r1o_scientific_erratum.md`, this report, and the
`r1o_r_*` artifacts.

## 29. What Did Not Change

No force model, no propagator, no light-time solver, no measurement model, no estimator, no
covariance method, no K solve-for mathematics. No file under `lunar_od/` was modified. No historical
number was deleted from any report.

## 30. Academic / Methodological Interpretation

The defect changed the six-state measurement Jacobian but not the K column. Because K identifiability
is *defined relative to the state-column subspace*, a wrong state Jacobian can create a completely
artificial appearance of parameter observability even when the parameter sensitivity itself is
numerically correct — and it can do so while every parameter-sensitivity check passes.

> **Parameter-sensitivity validation alone is insufficient for observability studies. The
> state-sensitivity basis against which the parameter is compared must be independently qualified.**

This project had qualified the K chain thoroughly — Phase 17A-R's four finite-difference oracles,
Phase 17-R's estimator integration, Phase 17-R1COV's exact rational covariance oracle. Every one of
those was correct, and none of them could have caught this, because none of them tested the *other*
side of the comparison.

A second, sharper lesson comes from s18: **a controlled comparison cannot detect a defect shared by
its control and treatment arms.** R1O-D built a careful bridge experiment, isolated a genuine
physical difference, and attributed to it an effect that was ~97-99% caused by a defect present in
both arms. The experiment was well designed for the hypothesis it tested and structurally blind to
the one that mattered.

This is a **reproducibility / methodological correction**, not a novel scientific discovery. Its
value lies in what it says about how observability studies should be qualified, not in the software
defect itself.

## 31. Operational / Software-Verification Interpretation

A navigation implementation can report a small, well-conditioned parameter covariance while using a
corrupted sensitivity basis. Nothing in the residuals would look wrong; nothing in the parameter
derivative would look wrong; the covariance would be numerically clean and confidently too
optimistic about *which* parameters the data can separate.

For operational OD:

- correct residuals alone are not enough;
- correct parameter sensitivity alone is not enough;
- **state-Jacobian consistency against independently propagated truth is also required.**

The specific trap here — a column-major/row-major mismatch at a module boundary — is a classic
MATLAB-to-Python porting hazard, and this repository carries exactly that lineage (the propagator's
docstring cites MATLAB's `Phi(:)`). It is worth checking anywhere a flattened matrix crosses an
interface. This is a software-verification lesson, not a new mission capability.

## 32. Regression Protection

Targeted re-runs, all green:

- `tests/test_r1o_stm_layout.py` — 5 new tests, pass; fail on mutation, pass on repair.
- `examples/phase17_r1o_r_stm_oracle.py` — `R1O_STM_LAYOUT_ORACLE_GATE = PASS`.
- `examples/phase17_r1od_campaign.py` (production DDOR, unmodified) — reproduces every published
  value, confirming the repair did not perturb production.
- `examples/phase17_r1o_r_requalification.py` — historical reproduction bitwise exact; K columns
  bitwise invariant across all four affected designs.

## 33. Full Suite

See s36's verdict block for the counts. Every non-pass was classified; none is attributable to this
phase.

## 34. What This Phase Actually Established

1. The repository's STM storage contract is column-major, established by measurement against an
   oracle that carries no variational block at all.
2. R1O's published surrogate results were generated with `Phi^T`, reproduced here exactly.
3. Corrected, neither R1O candidate rotates K_SRP's information direction; the landmark case is a
   large magnitude gain, the DDOR case is a direction *loss*.
4. R1O's ranking and candidate selection are withdrawn.
5. R1O-D's production result is valid; its causal attribution was wrong by a factor of ~30-100, and
   the reason it could not detect this is structural.
6. R1O-OPT's production result is valid and is now corroborated independently from the surrogate side.
7. All K columns, the range-only baseline, R1M, R1COV, and both production measurement models are
   unaffected.
8. Permanent, behaviour-based, mutation-verified regression protection now exists.

## 35. What Remains Unknown

- Whether weak K_SRP identifiability is specific to this synthetic lunar trajectory or general. This
  phase deliberately changed no geometry.
- Whether any observable class *not* yet examined could rotate K's direction. R1O's search was
  conducted with a corrupted state basis, so its *screening* — not only its ranking — is unreliable;
  candidates it discarded were discarded on bad evidence.
- The nonlinear-estimator behaviour of the corrected designs. Everything here is linearized
  information geometry.
- Whether the ~0.005 event-physics term isolated in s18 is stable across geometries.

## 36. Why Geometry Generalization Is Next

Geometry generalization will run many trajectory cases through the same analysis machinery. Had the
helper remained defective, a single coding error would have contaminated an entire parameter sweep —
and, as this phase shows, produced results that look scientifically coherent and internally
consistent while being wrong. Sweeping first and repairing second would have meant re-running
everything.

This is causal isolation, not bureaucracy: repair first, generalize second.

The next scientific question is no longer *"which additional observable should we try next?"* — both
selected R1O candidates have now been examined in production form, and the screening that selected
them is itself no longer trustworthy. It is:

> Is weak K_SRP identifiability specific to the current synthetic lunar trajectory, or does it
> persist across a broader class of lunar orbital and force geometries?

That motivates **Phase 17-GEO**, which this phase does not begin.

## 37. Final Verdict

See the verdict block below.

## 38. Exact Next Action

Owner review of this erratum and of the withdrawn R1O ranking. Then Phase 17-GEO
(geometry generalization). Not started here.
