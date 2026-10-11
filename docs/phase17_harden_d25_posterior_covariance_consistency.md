# Phase 17-HARDEN-D25 — Production posterior covariance consistency

- **Date:** 2026-10-10 – 2026-10-11
- **Branch:** local `phase17-harden-bls1` (worktree `python_port_phase17r0`), on top of BLS1 `742576f`.
  Repair and predeclaration: `aac72ed`.
- **Predeclaration:** `artifacts/phase17_d25_predeclaration.json`, committed with the repair before any
  qualification number was computed.
- **Evidence:** `artifacts/phase17_d25_qualification.json`, `artifacts/phase17_d25_e2e.json`,
  `examples/phase17_d25_qualification.py`, `tests/test_harden_d25_posterior_dynamics_contract.py`.

## 0. Final status

```
PHASE17_HARDEN_D25_GATE                 = FAIL as computed -> ADJUDICATED (pending owner acceptance)
                                          G1 G2 G3 G5 G6 G8 PASS; G7 reproduction PASS (bitwise);
                                          G4 (predeclared FD oracle) FAIL: its K and velocity steps sat in the
                                          FD noise floor (error ~1/h, measured); at steps out of the floor the
                                          analytic design matches FD to <= 3.8e-5 (state) / 6.2e-4 (K) per
                                          column and the oracle covariance to 1.6e-3 in sigma, vs 9.1 % for
                                          the pre-D25 posterior
POSTERIOR_COVARIANCE_MODEL_CONSISTENCY  = QUALIFIED: the posterior is linearised on the solve's dynamics
                                          (independent assembly 2.5e-13; KP2 solver 1e-6)
PRODUCTION_BLS_ESTIMATE_STATUS          = UNCHANGED (bitwise in all 8 tested configurations) / REQUALIFIED
                                          end to end (frozen KP2 job: <= 5.7e-5 sigma)
PRODUCTION_K_SIGMA_STATUS               = QUALIFIED: frozen KP2 sigma criterion 1.000000-1.000001 (was 0.917);
                                          caveat HARDEN-D28 (default weak prior, < 1e-5 on KP2)
KP2_D25_MATERIALITY                     = CONFIRMED_NOT_MATERIAL_TO_HEADLINE_CLASSIFICATIONS
```

KP2's frozen PRODUCTION_BLS gate now passes both criteria as computed, end to end, with the BLS1 + D25 estimator.
KP2's FD gate remains as previously adjudicated. Superseded uncertainty values: gravity-campaign stages G and I
(§4). No estimate changed anywhere. The common K level is not claimed.

## 1. What D25 was

The two-way-range estimators compute the estimate and the posterior covariance in two separate propagations:

- the iterations, which produce the estimate;
- one pass at the solution, which builds the design for the covariance.

The posterior pass did not receive the full dynamics contract the iterations used. Source audit (blast radius,
predeclared):

| estimator / path | solve dynamics | posterior dynamics before D25 |
|---|---|---|
| `estimate_two_way_range_bls_lm`, K solve-for | harmonic field (matched STM) + SRP(K) | SRP(K), **no harmonic field** — D25 as recorded in KP2 |
| `estimate_two_way_range_bls_lm`, six-state | harmonic field + fixed SRP | **neither** |
| `estimate_two_way_range_srif`, six-state | fixed SRP | **no SRP** (SRIF has no harmonic option) |
| `estimate_two_way_range_srif`, K solve-for | SRP(K) | SRP(K): consistent |
| position / range-rate estimators | no SRP or harmonic options | unaffected |
| any estimator with `srp=None` and no field | central / J2 | consistent; unchanged |

D25 is therefore broader than KP2 recorded. It also covers every six-state BLS solve with a harmonic field or a
fixed SRP, and every SRIF six-state solve with a fixed SRP, whenever a posterior was requested.

**Why the estimate was right while the uncertainty was not.** The estimate is the fixed point of the
iterations, and those always used the full model. The posterior pass runs after the solve and feeds nothing
back. It only re-linearises at the solution, and it did so on a different force model: the Jacobian of a
central/J2 (or SRP-free) trajectory, not of the trajectory that was fitted. Gate G2 confirms this. In all eight
fixture configurations, including the four D25 changes, the estimate, status, iteration count and final cost are
bitwise identical before and after the repair.

## 2. Repair (`aac72ed`, `lunar_od/estimators.py`, +45 / −0)

`_posterior_dynamics_kwargs(srp, harmonic_kwargs)` builds the posterior pass's dynamics arguments exactly as the
iterations build theirs: SRP when given; a harmonic field always with `harmonic_stm_opt_in=True`, the matched
analytic Pines gradient. The three posterior helpers accept these arguments.

- The BLS K and six-state branches pass the solve's `srp` and harmonic field.
- The SRIF six-state branch passes its fixed `srp`.

With nothing active the call is exactly the historical one. Untouched: the estimate path, weights, prior,
tolerances, covariance algebra (R1COV QR / six-state information), the force model, D27.

## 3. Qualification (gates frozen in `artifacts/phase17_d25_predeclaration.json`)

| gate | result | value |
|---|---|---|
| G1 default invariance | **PASS** | four default configurations (BLS and SRIF, six-state and K): posteriors bitwise identical to `742576f` |
| G2 estimate invariance | **PASS** | eight configurations, including the four D25 changes: estimate, status, iterations and cost bitwise identical; posteriors change only in the D25 configurations (up to 5.3 % in σ on the 3.3 h fixture) |
| G3 O1 independent assembly | **PASS** | KP2 HF K path: 2.5e-13 in σ, 9.9e-14 in correlation; stage F: ≤ 9.5e-10 |
| G4 O2 central FD of the nonlinear solve model | **FAIL as computed** | max σ 1.13e-2 (K), velocity 0.25–0.56 %; positive control PASS (pre-D25 K 7.8 %) |
| G5 frozen KP2 σ criterion | **PASS** | KP2 solver / repaired production σ = 1.000000–1.000001 (pre-D25 0.917 K, 0.972 z) |
| G6 KP2 classifications | **PASS** | no KP2 decision input reads the production posterior; KP2 tests pass |
| G7 blast radius | reproduction **PASS** | stored stage F and stage I covariances reproduced bitwise by the pre-D25 pass; statistics re-derived (§4) |
| G8 regression | **PASS** | 4 new tests; full suite on a clean worktree 1475 passed, 29 skipped, 0 failed |

End to end (`artifacts/phase17_d25_e2e.json`): the frozen KP2 `job_bls`, rerun unchanged with the repaired
estimator from the declared a priori, gives an estimate within 5.7e-5 σ and σ ratios of 1.000000–1.000001.
Both frozen criteria pass as computed.

### G4 adjudication (post-hoc, labelled; the predeclared result is unchanged)

O2 failed on every σ, systematically, while O1 and the independent KP2 solver agree with production to 1e-13 and
1e-6. Three labelled diagnostics locate the cause in the oracle:

1. **K step in the noise floor.** In simplified CANON the K-column FD error falls as ~1/h, both with
   CONICAL_PENUMBRA (3.8e-2 at h = 1e-4, 3.3e-3 at 1e-3, 3.9e-4 at 3e-3, 8.8e-5 at 9e-3) and with NO_SHADOW
   (4.1e-3 … 4.2e-5). That is the signature of integrator noise in the differenced residuals, not a wrong
   derivative. HARDEN-D05 documented the same FD-floor behaviour at the non-smooth penumbra right-hand side.
2. **HF with h_K = 5e-3 and 2.5e-3** (central steps keep K > 0; the two agree to 2.2e-3): the K σ
   disagreement drops from 1.1 % to 2.9e-4 – 4.1e-4. Velocity σ still differ by 0.25–0.33 %.
3. **HF velocity-step study** (all columns saved: `artifacts/phase17_d25_posthoc_fd_columns.npz`):

   | | FD vs analytic design column |
   |---|---|
   | position, step 1 m | 3.3e-6, 3.8e-5, 2.7e-5 |
   | velocity, step 1e-2 m/s | 1.3e-5, 1.2e-6, 3.2e-7 |
   | velocity, step 3e-4 m/s | 8.8e-4, 2.8e-5, 4.4e-5 (noise ~1/h again) |
   | K, step 5e-3 | 6.2e-4 |

   With the velocity step at 1e-2 m/s, the oracle covariance agrees with the repaired posterior to 1.55e-3 in σ
   (K 8.7e-4) and 1.0e-3 in correlation. The pre-D25 posterior differs from the same oracle by 9.1 % (K) and
   2.9 % (z).

The repaired design agrees with the FD of the solve's own nonlinear model to the oracle's own floor, column by
column. The remaining covariance-level difference is that floor amplified along the weak state/K directions. The
predeclared steps (1e-4 for K, 1e-3 m/s for velocity) were inside the noise floor. That is an oracle-design
error, recorded as such; the criterion is not relaxed. G4 stays FAIL as computed; the adjudication is pending
owner acceptance, as for D23 and KP2.

## 4. Blast radius: the gravity-fidelity campaign (stages F/G/I)

The high-order gravity covariance-realism campaign (`od_covariance_campaign`, stages F, G and I) used the six-state
BLS with GL1800F E50_G50 / E100_G100 and `return_posterior=True`. That is the six-state D25 branch.
`examples/phase17_d25_qualification.py --stage-f` reads its 40 stored solutions read-only and rebuilds the
posterior at each one. The rebuild uses the same observation realisation (seed), prior P0, weights and tolerances.

**The historical numbers carry D25.** The pre-D25 posterior reproduces every stored `P_hat` **bitwise**
(maximum difference 0.0). The repaired posterior agrees with the independent O1 rebuild to ≤ 9.5e-10 in σ and
≤ 3.6e-10 in correlation.

Stage G statistics, re-derived with stage G's own formulas:

| statistic | E100_G100 historical | **E100_G100 repaired** | E50_G50 historical | E50_G50 repaired |
|---|---|---|---|---|
| ANEES (95 % band 4.58–7.61) | 11.58 | **9.01** | 1370.6 | 1184.8 |
| overconfidence ratio | 1.93× | **1.50×** | 228× | 197× |
| inside the ANEES band | no | **no** | no | no |
| formal 3D position σ | 64.41 m | **63.84 m** | 64.41 m | 63.88 m |
| 95 % Mahalanobis coverage | 0.70 | 0.80 | 0.00 | 0.00 |
| per-axis 1σ coverage | 0.517 | 0.508 | 0.025 | 0.033 |
| mean formal σ change | — | +2.0, −1.0, −0.9 %; −1.1, −1.6, −1.1 % | — | +1.6 … −1.5 % |

Individual σ move by only 1–2 %, but the error vector is dominated by the weak, correlated directions, and NEES
there drops 22 %. **Superseded:** the G100 headline "1.93× overconfident (ANEES 11.58)" in stage G and in
`linearity_nonlinearity_validity_domain_qualification.md` §28; the replacement is **1.50× (ANEES 9.01)**.
**Not superseded:** the qualitative conclusions. G100's formal covariance is still overconfident and outside the
ANEES band, and G50 is still model-error dominated. The estimates, errors and holdout numbers are unchanged
(G2). Phase 6.2's "0.0219 σ" is in measurement-noise units and is unaffected. The Phase 6 discriminator records
single-realisation NEES values only as documentation, not for decisions.

**Stage I** (G100, 24 nonlinear cubature solves; `artifacts/phase17_d25_stage_i.json`). The stored per-point
`P_hat` are reproduced **bitwise** by the pre-D25 pass on the shared fit geometry. With the repaired posterior:

| statistic | historical | repaired |
|---|---|---|
| trace(P_form⁻¹ C_cub) (expect 6) | 5.96 | **4.06** |
| overconfidence | 0.99× | **0.68×** |
| generalized eigenvalues of C_cub vs P_form | 0.078, 0.168, 0.364, 0.432, 0.633, **4.282** | 0.078, 0.170, 0.366, 0.621, 1.091, **1.735** |
| formal 3D position σ | 64.41 m | 63.84 m |

The single strongly under-covered direction (4.28) was largely D25. The miscalibration stays directional
(eigenvalue spread ~22×), so `linearity_nonlinearity_validity_domain_qualification.md` §27–28 holds qualitatively.
Its numbers (ANEES 11.58, 1.93×, and stage I's 5.96 / 4.28) are superseded by the values here. Stage I's nonlinear
cubature now implies the formal covariance is *conservative* overall (0.68×), while stage G's 20-sample Monte Carlo
gives 1.50× overconfidence. Reconciling the two is a science question outside D25 and is not attempted.

## 5. KP2 materiality

**Not material to any KP2 headline classification**, confirmed in code and by tests:

- every KP2 decision quantity comes from the KP2 solver's own QR covariance (`phase17_kp2_core._gauss_newton`,
  `sol["cov"]`);
- the production posterior is read only by `job_bls` and the post-hoc stationarity check, i.e. only by the
  PRODUCTION_BLS verification gate;
- `tests/test_kp2_m3_recovery.py` passes unchanged.

That gate's σ criterion now passes as computed (G5 and end to end). `M3_VARIATION_PARTIALLY_RECOVERABLE` and the
unclaimed K level stand.

## 6. Separate observation: HARDEN-D28 (open, not changed)

With no explicit prior, `_prior_information_and_scale` supplies a weak default state prior (15 km, 5 m/s; none on
K). It enters the LM normal matrix and the reported posterior, but not the acceptance objective or the step's
right-hand side. The estimate is therefore the data-only least-squares solution, while the reported covariance
is that of the weakly regularised problem. This is undocumented, and it is a contract question separate from D25,
which concerns dynamics.

Measured effect on σ:

- 3.3 h single-station regression fixture: ≤ 1.4e-4;
- KP2 15-orbit arc: < 1e-5 (BLS1 diagnostic: KP2 / harmonic-consistent σ ratio 1.0000).

Changing it would change every default six-state posterior bitwise, so it is recorded, not repaired. The D25
qualification compares like with like by including this default prior in every rebuild of an estimator posterior.

## 7. Verification

- New tests: `tests/test_harden_d25_posterior_dynamics_contract.py` (4 tests, about 27 s).
  - The posterior-kwargs contract.
  - BLS K path with a degree-8 GL1800F field: production equals the rebuild to 8.7e-15 in σ; the historical
    construction is 4–5 % away.
  - Six-state BLS with field + SRP: information equals the rebuild to ≤ 1e-10; the historical construction is
    up to 2.6 % away in σ.
  - Six-state SRIF with SRP: ≤ 2e-17; historical 1.5e-6.
  - The estimate is unchanged with and without the posterior pass.
- Focused: `test_k_srp_*`, `test_estimator_lm_consistency`, HARDEN BLS1 tests: pass.
- Full suite on a clean detached worktree of `aac72ed` plus the uncommitted D25 files, with
  `LUNAR_OD_PROJECT_ROOT` = workspace root: **1475 passed, 29 skipped, 0 failed** (BLS1's 1471 + 4).
- The shared worktree also carries another session's uncommitted desktop-app and `examples/run_scenario_config.py`
  edits. They are not part of this phase and are not staged.

## 8. Not done

D27, D28, stochastic K, production M3 / counted-Doppler capability, new science. The common K level is not
claimed. The external `od_covariance_campaign` repository is read, never modified. Its stage F/G artifacts are
superseded by §4 here, not rewritten.
