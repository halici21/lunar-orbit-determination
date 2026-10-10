# SCIENTIFIC ERRATUM — PHASE 17-R1O

> ### ERRATUM (Phase 17-R1O-D2, enforced by Phase 17-HARDEN): lunar occultation
>
> **0.3204 is superseded.** The range + production ΔDOR 5 nrad value 0.3204 (and its surrogate twin 0.3203) is labelled
> **`EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE`**. FXR corrected its Earth epoch, but
> its ΔDOR rows were still accepted on station elevation alone, without a lunar-occultation test. As a result,
> **93 of 263 rows (35%) had the spacecraft behind the Moon** for at least one station.
>
> **Authoritative values** (Phase 17-R1O-D2, canonical Goldstone–Canberra, 5 nrad, both stations visible under the
> production elevation + occultation contract):
>
> | quantity | value |
> |---|---:|
> | range-only `f_perp` | **0.091508** |
> | range + ΔDOR `f_perp` | **0.172351** |
> | gain Δ`f_perp` | **+0.0808** (STRONG_DIRECTION_GAIN) |
> | σ_K/K | **7.46%** |
>
> **R1O surrogates under the same contract** (Phase 17-HARDEN, `artifacts/phase17_harden_visibility_requalification.json`):
>
> | surrogate | 5 nrad `f_perp` | 1 nrad `f_perp` | rows |
> |---|---:|---:|---|
> | ΔDOR-like | 0.1722 | 0.1381 | 170 kept of 263 |
> | Earth LOS | 0.0919 | — | 360 kept; 230 of 590 were occulted |
>
> The R1O-D-era bridge surrogate uses an Earth frozen at t = 0 and is historical only. On the legacy (elevation-only) path, every FXR number
> still reproduces bitwise. The single current status is in `docs/phase17_authoritative_status.md`. The defect is
> HARDEN-D04 in `docs/phase17_harden_defect_ledger.md`. The text below is kept unchanged as the historical record.


> ### SECOND ERRATUM (Phase 17-FXR) — Earth-epoch fixture
>
> §5–§8 below were computed on the published Earth fixture: the Earth position given to the range model came from an epoch **504000 s (5.83 days) earlier** than the Earth rotation, Sun and spacecraft state, and was **held constant after 28260 s (3.9986 orbits)**. On the consistent fixture, with the corrected STM:
> - range only: 0.0915
> - landmark 1 µrad: 0.2524
> - ΔDOR-like 1 nrad / 5 nrad: 0.3619 / 0.3203
> - Earth LOS: 0.0920
> - production ΔDOR 5 nrad: 0.3204 (STRONG_DIRECTION_GAIN)
>
> §8's "neither ... meaningfully rotates K_SRP's information direction" and §7's "R1O-D conclusion
> VALID" do not hold on the consistent geometry. §2–§4, the STM defect and its repair, stand.
> Full map: `docs/phase17_fxr_earth_epoch_fixture_erratum.md`; evidence: `docs/phase17_fxr_earth_epoch_fixture_requalification.md`.

**Status:** Phase 17-R1O's cross-observable ranking is **SUPERSEDED**.
**Issued by:** Phase 17-R1O-R (`docs/phase17_r1o_r_stm_layout_repair.md`).
**Repair commit:** `75ba49a7e86cf7c4ea01e45ca757126159dc0654`.
**Affected report:** `docs/phase17_r1o_additional_observable_feasibility.md`.

Phase 17-R1O's original numbers are deliberately preserved, in this document and in its own report.
They are not deleted. This erratum records what was published, what was wrong with it, what the
corrected analysis produces, and which later conclusions survive.

---

## 1. What R1O originally claimed

R1O reported that two new observables strongly rotate the K_SRP information direction away from the
six-state subspace, and ranked them as Tier-1 candidates:

| published claim | value |
|---|---:|
| `OPTICAL_BEST_F_PERP` (landmark LOS, 1 urad) | **0.8757** |
| `OPTICAL_BEST_FRACTIONAL_SIGMA_K` | **0.001939** |
| `OPTICAL_BEST_THETA_K` | **61.13 deg** |
| `DDOR_BEST_F_PERP` (1 nrad) | **0.4419** |
| `DDOR_BEST_FRACTIONAL_SIGMA_K` | **0.01224** |
| `BASELINE_F_PERP` (range only) | 0.2523 |
| `BEST_INFORMATION_DIRECTION_CLASS` | `STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION` |
| `PROMISING_OBSERVABLE_FOUND` | `YES` |

Its executive summary called the landmark result *"the single strongest result in the entire K_SRP
investigation to date"* and described the rise from ~15 deg to ~60 deg as *"a substantial rotation of
K's independent signature toward orthogonality with the six-state subspace, not a marginal one."*

## 2. The defect

`examples/phase17_r1o_core.py`, function `chain_to_augmented_columns`:

```python
phi = np.asarray(nom48_row[6:42], dtype=float).reshape(6, 6)     # C order - WRONG
```

`lunar_od.dynamics` packs and propagates Phi **column-major** (`reshape(-1, order="F")`,
`dynamics.py` s1249/s1292; the docstring at s386 states it explicitly: *"stored and flattened in
column-major order, matching MATLAB's `Phi(:)` convention"*). For a column-major flattening `v`:

```
v[i + 6j] = Phi[i, j]          (pack,  order="F")
M[a, b]   = v[6a + b]          (unpack, order="C")
          = Phi[b, a]     ==>  M = Phi^T
```

Phi is not symmetric, so this substituted **Phi^T for Phi** in every surrogate state-design row R1O
built. The repaired line is `reshape((6, 6), order="F")`.

## 3. Why the defect matters

Phi(t,t0) is defined by `dx(t) = Phi(t,t0) dx0`, and the measurement state Jacobian is
`H_x0 = H_x(t) Phi(t,t0)`. R1O's central metric

```
f_perp = ||b_K,perp|| / ||b_K||
```

measures the fraction of the K design column that the **six state columns cannot reproduce**. Using
Phi^T does not perturb that measurement slightly — it replaces the entire subspace K is being
compared against with a different one. A wrong state basis can therefore manufacture an arbitrary
appearance of parameter observability while the K sensitivity itself remains numerically perfect.

That is exactly what happened: R1O's K columns were correct throughout.

## 4. Affected and unaffected

Verified by exhaustive search of every `reshape(6,6)` in the repository, not assumed.

**Affected** (all consumed the defective helper):
- `phase17_r1o_core.build_landmark_arc` — R1O's landmark surrogate
- `phase17_r1o_core.build_ddor_arc` — R1O's DDOR-like surrogate
- `phase17_r1o_celestial` — Earth-center LOS negative control
- `phase17_r1od_surrogate_bridge` — the **surrogate side only** of R1O-D's bridge

**Unaffected** (verified, in several cases by re-running them):
- R1O's range-only baseline — built by the production `two_way_range_nominal_and_initial_jacobian`
  and `_two_way_range_k_srp_column`, which use `order="F"`
- Production DDOR (`lunar_od/delta_dor.py`) — re-run in full; reproduces every published value
- Production optical (`lunar_od/lunar_landmark_optical.py`) — its callers pass an already-unpacked
  `order="F"` Phi, and a transpose-sensitivity test already guards it
- **All K columns everywhere** — `S_K` is a plain 6-vector in columns `[42:48]`, never reshaped, so
  no storage-order convention applies to it. Verified bitwise identical before and after repair.
- R1M, R1COV — no use of the helper

## 5. Corrected numbers

One arc, R1O's own published configuration (15 orbits, three DSN stations, 90 s cadence, 782 range
observations). Identical landmark set, station pair, epochs, noise, K truth and sampling in both
columns — **only Phi's storage order differs**.

| case | `I(K\|x)` | `f_perp` | `theta_K` | `sigma_K/K` |
|---|---:|---:|---:|---:|
| range only (production Jacobian, unaffected) | 1.162817e+06 | 0.252292 | 14.61 | 9.2735% |
| landmark 1 urad — **historical (Phi^T)** | 2.658848e+09 | **0.875701** | 61.13 | 0.1939% |
| landmark 1 urad — **corrected (Phi)** | 2.218707e+08 | **0.252964** | 14.65 | 0.6714% |
| DDOR 1 nrad — historical | 6.675259e+07 | **0.441901** | 26.23 | 1.2240% |
| DDOR 1 nrad — corrected | 1.681884e+06 | **0.070144** | 4.02 | 7.7108% |
| DDOR 5 nrad — historical | 4.119144e+06 | 0.363285 | 21.30 | 4.9272% |
| DDOR 5 nrad — corrected | 1.337112e+06 | 0.206980 | 11.95 | 8.6480% |
| Earth LOS 1 urad — historical | 1.250870e+06 | 0.261667 | 15.17 | 8.9412% |
| Earth LOS 1 urad — corrected | 1.214443e+06 | 0.257829 | 14.94 | 9.0743% |

The historical column reproduces R1O's published values **exactly** — `0.875701` against a published
`0.8757`, `0.441901` against `0.4419`, `I(K|x) = 2.658848e+09` and `6.675259e+07` against the same
digits in R1O's own verdict block. The requalification is therefore genuinely re-running the path
that produced the published result, not a lookalike.

## 6. Corrected scientific interpretation

**The landmark headline collapses.** At 1 urad, corrected `f_perp = 0.2530` against a range-only
baseline of `0.2523` — a gain of **+0.0007**, not the published rise to `0.8757`. The published
"rotation from 15 deg to 61 deg" was `14.61 deg -> 14.65 deg`.

**`sigma_K/K` still improves, and this is the distinction that matters.** Corrected landmark
observations drive fractional K uncertainty from 9.27% to 0.67%. They add a great deal of K
information *magnitude* while rotating its *direction* almost not at all. R1O's own framework
already contained the vocabulary for this (`I_K|x` vs `f_perp`); the defect simply made a
magnitude-only result look like a direction result.

**DDOR is worse than its published claim, and worse than the baseline.** Corrected `f_perp` at
1 nrad is `0.0701`, *below* the range-only `0.2523`. Adding a strongly-weighted observable whose K
signature lies largely inside the state subspace raises `I_KK` faster than `I_K|x`, so the *fraction*
falls even as total information rises.

**The Earth-LOS negative control survives.** `0.2617 -> 0.2578`, still negligible against a `0.2523`
baseline. Numbers changed slightly; the scientific classification did not.

## 7. Downstream phases

**Phase 17-R1O-D (production DDOR) — conclusion VALID, causal explanation REVISED.**

R1O-D's own result is untouched and was re-run to confirm it: production DDOR is `MAGNITUDE_ONLY`
with `f_perp` below baseline at every tested precision (1 nrad `0.0738` ... 100 nrad `0.2641`). That
finding stands completely.

But R1O-D's *explanation* for why its production model disagreed with R1O's surrogate does not:

| noise | historical surrogate | corrected surrogate | production | STM-defect share |
|---:|---:|---:|---:|---:|
| 1 nrad | 0.441901 | 0.070144 | 0.0738 | **99.0%** |
| 5 nrad | 0.363285 | 0.206980 | 0.2121 | **96.8%** |

R1O-D attributed the whole surrogate-vs-production gap to differential light-time discarded by the
surrogate. In fact the corrected surrogate lands almost exactly on production: the STM defect
accounts for ~97-99% of the gap, and the genuine measurement-physics difference for the small
remainder — which acts in the **opposite** direction, making production marginally *better* than the
corrected surrogate rather than worse.

The methodological reason R1O-D could not see this is worth recording: its bridge experiment varied
one factor (event physics) between two arms that **both** used the defective Phi. A controlled
comparison cannot detect a defect its control and treatment arms share.

**Phase 17-R1O-OPT (production optical) — VALID.** Its production result never used the defective
helper, and it is the phase that discovered the defect. Its `MARGINAL_DIRECTION_GAIN` classification
is confirmed and now explained: the corrected landmark surrogate and the production optical model
agree that landmark observations are a magnitude, not a direction, improvement.

**Phases 17-R1M and 17-R1COV — VALID.** Neither uses the helper.

## 8. What the corrected R1O actually established

Not *"two strongly complementary observables found."* Rather:

> On this trajectory, neither a DDOR-like plane-of-sky angle nor a lunar-landmark line of sight
> meaningfully rotates K_SRP's information direction out of the six-state subspace at any tested
> precision. Both add information *magnitude*; the landmark case adds a great deal of it. The
> range-only baseline's `f_perp ~ 0.25` is not improved upon by either candidate.

`R1O_ORIGINAL_RANKING_VALID = NO`. The DDOR-over-landmark ranking R1O published rested entirely on
transposed state Jacobians, and both candidates' corrected direction metrics now sit at or below the
baseline they were supposed to beat.
