# F3 WP3.2 S3 — Earth's shadow on the spacecraft as an opt-in production force term (REQ-DYN-007, DYN-17)

- **Date:** 2026-11-26 (gate runs 2026-11-25/26)
- **Predeclaration:** `studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json`, committed in `a5954f1`
  before any S3 code, test or number. Two errata before any result (`6ebbed7`: the E_G2 Sun direction is drawn near
  the Earth direction, not the anti-Earth direction; `15468af`: the E_G3(a) small-angle precondition scales both cap
  radii and the separation), recorded in the predeclaration's `errata_before_any_result`.
- **Amendments** (`docs/assurance/amendment-log.md`, all after a result, all exploratory):
  - E_G3(a) FAIL and the exploratory set E_G3(a)-A, committed with its script in `222f160` before it ran;
  - E_G5(a) FAIL, E_G6 FAIL, the post-result epoch fix and the exploratory set E_G6-A, committed with its script in
    `4d12c59` before it ran;
  - the E_G3(a)-A/E_G6-A results and the strict xfails (`bcab6c9`).
- **Code:**
  - S3a (`665d483`): the new module `src/lunar_od/earth_shadow.py` and the term in the propagators and the D2 shadow
    events of `src/lunar_od/dynamics.py`;
  - S3b (`bcab6c9`): its transport through `src/lunar_od/propagation_contract.py` and `src/lunar_od/estimators.py`;
  - re-freeze (`855d504`): new table `F3_WP3_2_S3_REFROZEN_CURRENT_TREE_SHA256` first in
    `expected_current_tree_sha256`, `docs/development/path-migration.json` history extended, history-chain tests
    extended.
- **Tests:** `tests/test_wp3_2_earth_shadow.py` (E_G1(a), E_G2–E_G8), `tests/test_wp3_2_earth_shadow_transport.py`
  (EB_G1–EB_G3 and the re-freeze record).
- **Results:**
  - `results/wp3_2_s3_gate_values.json`: E_G2–E_G7, EB_G3 and cost, at `bcab6c9`, clean `src`
    (script `scripts/wp3_2_s3_gate_values.py`);
  - `results/wp3_2_s3_default_identity.json` (E_G1(b)) and its four inputs in `results/wp3_2_s3_default_identity/`:
    two runs before the change at `15468af` (clean), and the omit and `None` runs after it, started at `bcab6c9`
    and recording `855d504` (the re-freeze commit landed while they ran; `src` is the same in both, clean);
  - `results/wp3_2_s3_eg3a_oracle_diagnostic.json` (exploratory E_G3(a)-A; run before the S3 src was committed, on
    the uncommitted tree recorded in its `status`; the oracle and the production module are those committed in
    `665d483`);
  - `results/wp3_2_s3_eg6_floor_diagnostic.json` (exploratory E_G6-A; at `4d12c59`, with the uncommitted src fix that
    was committed unchanged in `665d483`).
- **Production defaults changed:** none. With `earth_shadow` omitted or `None`, all 79 recorded configurations
  are byte-identical before and after the change (E_G1(b)).

## 0. Final status

```
S3 (as predeclared)    = NOT QUALIFIED: E_G3(a), E_G5(a) and E_G6 FAIL; every other gate PASS
  E_G1 identity        = (a) the wrapper returns the RHS itself when off; (b) 79/79 configurations byte-identical
  E_G2 WP3.1 parity    = bit for bit, 0 mismatches in 2000 samples (463 lit, 729 penumbral, 808 umbral)
  E_G3 accuracy        = (a) FAIL: oracle precondition 1.41e-8 (> 1e-9); production vs oracle 2.97e-6 (<= 2.5e-4)
                         (b) SPICE gfoclt E2 4.8e-7 s, E3 3.6e-7 s (<= 0.01 s), same window counts
  E_G4 gradient        = 1.1e-7 (<= 1e-6) for grad nu_E and d(da)/dr, NO_SHADOW and CONICAL (714 / 710 samples)
  E_G5 variational     = (a) FAIL: STM 6.3e-8 (<= 1e-7) but the control without the Earth-shadow gradient is
                         3.7e-7, 5.9x (>= 10x required); (b) K sensitivity 1.8e-7 (<= 1e-6), control 0.90 (5e6x)
  E_G6 path agreement  = FAIL in all five cases (table in section 3)
  E_G7 shadow events   = PASS: 17 cuts with the Earth shadow (4 Earth), 13 without (0 Earth); every cut within
                         1.9e-6 s of a contact; lunar cuts move by up to 8.4 ms; the control raises;
                         Earth-only segmentation (NO_SHADOW) 6 Earth cuts
  E_G8 fail-closed     = PASS
  EB_G1-EB_G3          = options; transport through every propagation of the six public estimators; recovery
                         1.0e-7 m and 3.7e-10 m/s (BLS-LM), 1.1e-7 m (SRIF), against 0.268 m without the Earth
                         shadow (2.5e6x)
  E_G9 regression      = section 6
E_G3(a)-A (exploratory) = A1, A2, A3 PASS: the precondition failure is the oracle's absolute quad tolerance
E_G6-A (exploratory)    = A1 FAIL, A2 PASS: without shadow events the Earth shadow raises the path disagreement
                          3.4-14x above the Earth-shadow-free floor; with events it stays at the floor
Decision                = the owner's (reject_if: not offered for merge until a fix passes a recorded gate set)
REQ-DYN-007             = still partial: S4 SRP_LIGHT_TIME, S5 hold-out replication
```

## 1. What changed

- `EarthShadowOptions(radius_m=6378137.0, provenance=...)` (keyword-only, validated) selects the term. It needs an
  active `SRPOptions` (ValueError otherwise), and works with `NO_SHADOW` and `CONICAL_PENUMBRA` SRP alike.
- The model is WP3.1's, unchanged: conical shadow of a spherical Earth, the illuminated fraction nu_E from the disks
  on the sky (`asin(R/d)`), an early exit nu_E = 1 when the spacecraft is on the Sun's side of the Earth, the lunar
  and Earth fractions multiplied. The increment added to the force model is `(nu_E - 1) a_SRP` with `a_SRP` the
  production SRP acceleration including the lunar shadow; E_G2 shows it equals `wp3_1_force_terms` bit for bit.
- Variational terms: `d(da)/dr = (nu_E - 1) G_SRP + a_SRP (x) grad nu_E` in the 42- and 48-state blocks and
  `(nu_E - 1) g_SRP` in df/dK. The 6-state path never evaluates a gradient.
- D2 shadow events: with `shadow_events` and the Earth shadow on, the Earth's outer and inner contact functions join
  the pre-pass events and the contact check (alone with `NO_SHADOW` SRP).
- Every propagation entry point (`propagate_state`, `propagate_augmented_state`, the harmonics path,
  `propagate_state_with_k_sensitivity`, `propagate_truth_with_ephemeris`, `shadow_contact_times`) and the six public
  batch estimators take `earth_shadow`; `model_options`/`propagator_options` carry it only when given.
- Not transported (predeclared `design.deferred`): UKF, EKF, scenario config, CLI, desktop, observability.
- Post-result fix (amendment log 2026-11-26): with the Earth shadow on, the Earth lookup is wrapped in a per-epoch
  cache; the wrapper now carries the `MoonCenteredEphemeris.et0_s` that `lunar_j2_epoch` reads, without which the D2
  pre-pass on the default grail_pa_de440 J2 raised ValueError (E_G6(v)). The off path is untouched.

## 2. Size and cost of the term

- 6 h effect on the 6-state positions at K = 0.05: 0.56 m on the TERM orbit (cases i-iii), 9.1 m on the PLANE orbit
  with lunar eclipses (iv), 8.9 m on the real E2 eclipse (v). The predeclared expectation was 10-40 m.
- OD impact (EB_G3): solving without the term on Earth-shadowed truth biases the epoch position by 0.27 m.
- Cost per wrapper evaluation inside an eclipse: 93 us (value, 6-state), 270 us (value and gradient, 48-state).

## 3. Gates that failed

**E_G3(a).** The oracle's small-angle precondition gave 1.41e-8 against 1e-9 while the gated comparison was
2.97e-6 against 2.5e-4. Hypothesis recorded with the failure: the predeclared absolute quad tolerance (1e-15 sr) is
~1.5e-5 of the solar cap when every angle is scaled by 1e-3. E_G3(a)-A: with the tolerance relative to the cap
(1e-13 x cap) the precondition is 2.9e-12 (A1), production against that oracle 2.97e-6 (A2), the two oracles agree
to 3.8e-14 on the unscaled samples (A3). The predeclared result stays FAIL.

**E_G5(a).** The gated STM error passes (6.3e-8 <= 1e-7), but the negative control without the Earth-shadow gradient
is 3.7e-7, only 5.9x the gated error (>= 10x required). The predeclaration had named a real chance of a ratio below
10. No diagnostic.

**E_G6.** 6- vs 42/48-state positions (max over 42, 42 reference, 48), Phi relative to max|Phi|, and the Earth-shadow
effect on the 6-state positions; thresholds 1e-5 m, 1e-8, > 1 m:

| case | positions (m) | Phi | effect (m) | failing check |
|---|---|---|---|---|
| (i) TERM, J2, NO_SHADOW | 1.15e-5 | 1.5e-12 | 0.56 | positions, effect |
| (ii) + S1 tide | 2.5e-6 | 2.0e-11 | 0.56 | effect |
| (iii) zonal harmonics | 3.0e-6 | 8.0e-13 | 0.56 | effect |
| (iv) PLANE, CONICAL, events | 1.004e-5 | 1.9e-12 | 9.1 | positions |
| (v) real E2, GRAIL J2, events | 1.28e-5 (after the fix; ValueError before) | 5.3e-11 | 8.9 | positions (and the error) |

E_G6-A, the same cases with the Earth shadow off (`studies/phase17/results/wp3_2_s3_eg6_floor_diagnostic.json`):

| case | on (m) | off (m) | on/off |
|---|---|---|---|
| (i) | 1.15e-5 | 8.2e-7 | 14.1 |
| (ii) | 2.5e-6 | 6.6e-7 | 3.7 |
| (iii) | 3.0e-6 | 8.8e-7 | 3.4 |
| (iv) events | 1.0e-5 | 2.1e-5 | 0.47 |
| (v) events | 1.3e-5 | 2.2e-5 | 0.58 |

A1 (on <= 2x off in every case) FAILS: the integration-floor hypothesis is refuted for the unsegmented cases. With
shadow events (iv, v) the Earth shadow adds nothing above the floor. Untested hypothesis: without events the three
propagators step across the C1 kinks of nu_E at the Earth's penumbra boundaries with different step sequences. The
practical limit: with the Earth shadow on, eclipse arcs should be propagated with shadow events
(`PropagationProfile()`), as WP3.1's A2 and assumptions item 17 already require for the lunar shadow.

The three failing tests are strict xfails (as S2's L_G6): the predeclared assertions run unchanged, and a pass
fails the suite.

## 4. Expectations against outcomes

| gate | expected | computed |
|---|---|---|
| E_G2 | exact | exact |
| E_G3(a) | 1e-5..5e-5 | 3.0e-6 (precondition failed) |
| E_G3(b) | <= 1e-3 s | <= 4.8e-7 s |
| E_G4 | 1e-9..1e-8 | 1.1e-7 (the FD step's truncation) |
| E_G5(a) | 2e-8..5e-8, control 10..50x | 6.3e-8, 5.9x |
| E_G5(b) | 1e-8, control O(0.1) | 1.8e-7, 0.90 |
| E_G6 | 1e-7..2e-6 m, effect 10-40 m | 2.5e-6..1.3e-5 m, effect 0.56-9.1 m |
| E_G7 | 4-8 Earth cuts, 5-10 ms | 4, 8.4 ms |
| EB_G3 | 1e-5 m, 0.1-10 m without | 1.0e-7 m, 0.27 m |

## 5. Notes on passing gates

- E_G4: 79 conical samples have the spacecraft in the lunar umbra, where both the Earth-shadow gradient and its
  finite difference are exactly zero; the relative error is 0/0 (a RuntimeWarning) and `max` keeps the finite value.
  The production gradient is exactly zero there, so this hides no error, but the test does not count them.
- E_G7 information (not gated): the D2 position error at rtol 1e-11 against 1e-13 on the same cuts is 9.5e-5 m with
  the Earth contacts and 8.0e-5 m with the lunar cuts only.

## 6. Regression (E_G9)

See the PR for the full-suite run; `tools/check_repository.py` and the full pytest run are reported there with
their counts, as computed.

## 7. Agent decisions

As predeclared (`agent_decisions`), plus, after the results: the epoch fix in `_share_earth_shadow_lookup` (a
defect, not a gate change), the strict xfails, and the two exploratory diagnostics. All flagged for the owner's
review.

## 8. Reproduce

```text
python -m pip install -e .
python -m pytest tests/test_wp3_2_earth_shadow.py tests/test_wp3_2_earth_shadow_transport.py -q
python studies/phase17/scripts/wp3_2_s3_gate_values.py LABEL OUT.json
python studies/phase17/scripts/wp3_2_s3_default_identity.py run omit OUT.json
python studies/phase17/scripts/wp3_2_s3_eg3a_oracle_diagnostic.py OUT.json
python studies/phase17/scripts/wp3_2_s3_eg6_floor_diagnostic.py OUT.json
```
