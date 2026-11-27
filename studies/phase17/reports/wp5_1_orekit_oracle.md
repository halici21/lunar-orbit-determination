# F5 WP5.1 — The Orekit external oracle made permanent (AC-3, R-18)

- **Date:** 2026-11-26 (generation and gate runs 2026-11-25/26)
- **Predeclaration:** `studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json`, committed and pushed in
  `a56c08b` before any WP5.1 run.
- **Amendment:** one, before any cross-tool number: the time-aligned adapter integrates the production 6-state
  RHS instead of the reference kernel `f3body_moon` (`docs/assurance/amendment-log.md`, 2026-11-25; commit `8324e50`).
- **Generators:** `tools/oracle/` (`8324e50`); fixtures `tests/fixtures/external_oracle/` (`3dffec3`, made byte-exact in
  the next commit), generated with the tools at `8324e50`.
- **Test:** `tests/test_external_oracle.py` (no Orekit).
- **Results:**
  - `studies/phase17/results/wp5_1_orekit_oracle.json` (every gate and information item; script
    `scripts/wp5_1_orekit_oracle_gates.py`);
  - `studies/phase17/results/wp5_1_link_diagnostic.json` (attribution of O_L, information; script
    `scripts/wp5_1_link_diagnostic.py`).
- **Documentation:** `docs/validation/external_cross_validation.md` (site page "External cross-validation").
- **Production code changed:** none.

## Status

**AC-3 partially met, as computed.**

| gate | result |
|---|---|
| O_P preconditions (a)–(f) | (a)(b)(c)(e)(f) pass; **(d) fail** (126 ns over the day > 50 ns). The predeclared remedy (pinned Orekit time map) was applied |
| O_D dynamics < 1 mm/day | **pass, 5/5** (max 0.115 mm, D3) |
| O_L production = aligned force model ≤ 0.1 mm | 3/5 pass; **D2-50 (0.64 mm) and D2-100 (1.71 mm) fail**: production integrator, no step cap (§4) |
| O_M2 two-way range < 1 mm | **pass, 2/2** (Moon-centred 5.8 µm, barycentric 0.13 mm) |
| O_M3 counted Doppler < 1 µm/s | Moon-centred pass (0.040 µm/s); **barycentric fail (3.8 µm/s)**: attributed to the oracle's double-precision noise (§5) |
| L_M3 production counted Doppler = range form ≤ 1e-8 m/s | pass (8.9e-10 m/s) |
| O_F fixture integrity | pass (no hash mismatch) |
| O_R regression / docs | `tools/check_repository.py`, `tools/build_docs.py --strict` and the full suite: §7 |

The three failing cases are strict xfails in `tests/test_external_oracle.py`. Each names this report. No
threshold was changed, and no protected source file was edited.

## 1. What was built

The 2026-10-21 Orekit/GMAT trial lived outside the repository (`agent-sandbox/orekit-gmat-oracle`). WP5.1 brings
its Orekit part in:

- **Generator scripts** (`tools/oracle/`): the trial's Orekit setup, with the physics unchanged.
  - Moon-centred ICRF frame, GM override, MOON_PA_DE440 from the PCK records by jplephem with a Java provider.
  - The GL1800F provider.
  - The three Java helpers.
  - Added: a ladder that fixes each tool's integrator settings by self-convergence, measurement generation on
    shared inputs, a pinned time map, and a provenance writer.
- **Pinned fixtures:** five Orekit trajectories (600 s samples), the Orekit time map, the shared spacecraft
  trajectory, Orekit's station track, Orekit's measurements, the scenario, the lunar_od settings, and
  `provenance.json`.
  - The provenance holds the versions (orekit-jpype 13.1.9.0, JPype 1.7.1, JDK 25, Orekit 13.1.9 jars) and the
    SHA-256 of every input: an orekit-data listing digest, the JPL DE440/DE421 binaries, the SPICE kernels, GL1800F
    and the Java sources.
  - It also holds the SHA-256 of every fixture.
  - The fixtures are stored with `-text` (`.gitattributes`), so the hashes hold on any checkout.
- **Orekit-free test:** imports neither Orekit nor JPype (checked by AST). Its thresholds are checked against the
  predeclaration text.

The oracle environment was the trial's own (Orekit 13.1.9, JDK 25, the vendored orekit-data). Nothing was
downloaded.

## 2. Alignment as run

As predeclared (A_TIME … A_STATION), with two notes.

- **Precondition O_P(d) failed:** Orekit's TDB−TT model drifts 126 ns over the day relative to SPICE's (mean offset
  1.19 µs), above the 50 ns bound. As the predeclaration prescribes, the TT-aligned adapter uses Orekit's own
  mapping, pinned at the 600 s samples (`orekit_time_map.csv`, inverted by cubic splines). Its size is reported as
  `I_TIMEMAP`.
- **Amendment (adapter kernel):** the adapter integrates `dynamics._build_state_rhs`, the RHS that `propagate_state`
  integrates. It uses the numba fast path and the exact-epoch DIRECT MOON_PA_DE440 rotation for D2. The
  predeclared kernel was `f3body_moon`, whose Python harmonics cost ~0.13 s per call at 100×100. The two agree
  to 1.5e-16 relative in acceleration (D1a, D2-100, D3), and `f3body_moon` is still reported (`I_KERNEL`, D1/D3).

**Deviation in file layout, not in any gate:** the Java helpers are in `tools/oracle/jhelpers/`, not
`tools/oracle/java/`. A `java` package on `sys.path` hides JPype's `java.*` import hook.

## 3. Integrator settings (self-convergence, before any cross number)

Ladder rule: accept the first rung whose next rung changes the position by ≤ 0.1 mm over the day.

| case | Orekit DP853 (abs 1e-8 m, rel 1e-14) | lunar_od DOP853 (rtol 1e-13, atol 1e-9) |
|---|---|---|
| D1a | max step 300 s (300 → 60 s: 0.00037 mm) | no cap (→ 60 s: 0.0010 mm) |
| D1b | max step 300 s (0.0011 mm) | no cap (0.0025 mm) |
| D2-50 | max step 300 s (0.00096 mm) | **20 s** (no cap → 60 s: 1.20 mm; 60 → 20 s: 0.33 mm; 20 → 10 s: 0.0004 mm) |
| D2-100 | max step 300 s (0 mm) | **20 s** (no cap → 60 s: 1.22 mm; 60 → 20 s: 4.19 mm; 20 → 10 s: 0.0024 mm) |
| D3 | max step 10 s (20 → 10 s: 0.29 mm; 10 → 5 s: 0.075 mm) | **2 s** (rungs 118, 4.9, 4.9, 0.33, 0.26, then 2 → 1 s: 0.016 mm) |

## 4. Dynamics (O_D, O_L)

145 samples, 600 s apart, over one day. Distances are maxima over the day.

| case | O_D: lunar_od (TT, Orekit time map) − Orekit | end RTN (m) | O_L: production `propagate_state` − adapter (TDB) | runtime (gate script) |
|---|---|---|---|---|
| D1a | 1.96e-6 m — pass | (3.7e-10, −1.9e-6, 8.6e-10) | 2.97e-7 m — pass | 9.7 s |
| D1b | 1.28e-6 m — pass | (1.2e-8, −1.3e-6, 5.3e-10) | 4.11e-7 m — pass | 7.3 s |
| D2-50 | 1.66e-6 m — pass | (1.8e-7, −1.6e-6, 6.0e-9) | **6.38e-4 m — FAIL** | 73 s |
| D2-100 | 1.14e-5 m — pass | (1.6e-7, −1.1e-5, 3.4e-8) | **1.71e-3 m — FAIL** | 91 s |
| D3 | 1.15e-4 m — pass | (−2.3e-6, −1.1e-4, −1.5e-7) | 9.59e-6 m — pass | 587 s |

The O_L comparison runs the production path at its own setting. That is DOP853 at rtol 1e-13 / atol 1e-9 with
no step cap, because `propagate_state` exposes none. For D3 this still passes, because the production path
stops at the shadow boundaries itself (`shadow_events=True`).

Information items (`wp5_1_orekit_oracle.json`):

- **I_TIME:** integrating in TDB rather than TT moves the end state by 25.63–25.66 mm, almost entirely
  along-track. This is the term the alignment removes.
- **I_TIMEMAP:** SPICE's TDB−TT instead of Orekit's moves the TT trajectory by 0.176–0.182 mm. With SPICE's
  map, O_D would be 0.15–0.18 mm, which still passes.
- **I_KERNEL:** the reference kernel `f3body_moon` minus the production RHS, both integrated: 6.6e-7 m (D1a),
  2.2e-7 m (D1b), 6.7e-6 m (D3). D2 was not run, as the amendment explains.
- **I_LINK_direct:** production with the exact-epoch DIRECT rotation minus the adapter: 0.74 mm (D2-50),
  1.16 mm (D2-100).
- **I_TRIAL:** the Orekit trajectories regenerated here minus the 2026-10-21 trial's stored ref3 trajectories:
  0 m (D1a, D2-50, D2-100) and 2.6 mm (D3). D3's integrator setting differs: the trial's ref3 against the
  10 s step accepted here.

**Attribution of the O_L failures** (`wp5_1_link_diagnostic.json`, information):

| case | production − adapter, both uncapped | adapter uncapped − adapter accepted (20 s) | production DIRECT − production grid |
|---|---|---|---|
| D2-50 | 0.27 mm | 0.39 mm | 0.12 mm |
| D2-100 | 4.91 mm | 6.53 mm | 0.56 mm |
| D3 | 182 mm | 182 mm (adapter at 2 s) | — |

At the same uncapped setting, the production path and the adapter differ by up to 4.9 mm. Each also differs
from the converged solution by millimetres. So the O_L failure is the uncapped DOP853 at rtol 1e-13 not
converging with degree-50/100 harmonics; neither the force model nor the 60 s rotation grid explains it (the
grid term is 0.12–0.56 mm). The step size the error control chooses is too large for the harmonic field's
short-period content: the lunar_od ladder needs a 20 s cap, while Orekit's DP853 with its absolute tolerance
needs none.

**Finding for WP3.3 (long-arc integrator certification), not fixed here** (`dynamics.py` is protected):
`propagate_state` at rtol 1e-13 with no step cap carries 0.6–1.7 mm per day of integration error with GL1800F
50/100. The production default profile (VODE Adams 1e-11) was not part of this gate. For D3 the uncapped
adapter is 182 mm off its converged solution, while the uncapped production path agrees with that solution
to 9.6 µm. The cause of the adapter's D3 error was not investigated: it is not a production path, so this is
information only.

## 5. Measurements (O_M2, O_M3, L_M3)

120 epochs, 60 s apart, from 1 h to 3 h, at Canberra. Both tools use the same spacecraft trajectory (D1a,
pinned every 10 s) and Orekit's station track. Light time is Newtonian, with no media and no delay. Orekit was
given the state at the bounce epoch; the iteration needed at most 2 passes, with a final update of 1.4e-12 s.

| quantity | max abs | rms | mean | gate |
|---|---|---|---|---|
| two-way range, Moon-centred | 5.84e-6 m | 3.83e-6 m | 1.0e-7 m | O_M2 — pass |
| two-way range, barycentric | 1.34e-4 m | 5.05e-5 m | 1.41e-5 m | O_M2 — pass |
| counted Doppler, Moon-centred | 3.97e-8 m/s | 8.1e-9 m/s | 1.2e-10 m/s | O_M3 — pass |
| counted Doppler, barycentric | 3.82e-6 m/s | 1.33e-6 m/s | −1.7e-9 m/s | **O_M3 — FAIL** |
| one-way range, Moon-centred (information) | 5.60e-6 m | 3.84e-6 m | −1.9e-7 m | — |
| production `counted_doppler` − range form | 8.9e-10 m/s | | | L_M3 — pass |

The barycentric minus Moon-centred two-way range is 1.83–1.94 m in Orekit (precondition f), and lunar_od agrees
with it to 0.13 mm.

**Attribution of the barycentric Doppler failure (information, computed after the gate).** Each series was
fitted with a local degree-8 polynomial over 15 epochs. The rms residual about that fit measures the
sample-to-sample noise:

| series | lunar_od | Orekit |
|---|---|---|
| two-way range, Moon-centred | 1.88e-5 m | 1.86e-5 m |
| two-way range, barycentric | 1.87e-5 m | **3.83e-5 m** |
| counted Doppler, Moon-centred | 5.46e-7 m/s | 5.44e-7 m/s |
| counted Doppler, barycentric | 5.47e-7 m/s | **1.31e-6 m/s** |

The lunar_od noise is the same in both frames. Orekit's doubles in the barycentric frame, where its positions
are about 1.5e11 m (double-precision spacing about 30 µm). Counted Doppler divides a two-way range difference
by 60 s, so a range noise of 38 µm becomes a Doppler noise above 1 µm/s. The mean difference is −1.7e-9 m/s.
The failure is therefore attributed to the oracle's barycentric numerical floor, not to a light-time
difference. The attribution was made after the gate result, so it is an explanation, not a pass. The gate
stays failed (strict xfail); passing it would need a better oracle representation (for example
Moon-relative differences inside Orekit), which is left for a later work package.

**I_STATION:** lunar_od's own station chain (SPICE ITRF93) minus Orekit's station track is 0.224–0.258 m.
This is the Earth-orientation realization difference handled by WP2.4. It is information, not a gate.

## 6. Tests

`tests/test_external_oracle.py`:

- **Default (fast) tier:** the O_F and O_P checks, O_D/O_L for D1a and D1b, and the measurement gates.
- **Slow tier** (`LUNAR_OD_RUN_SLOW_TESTS=1`): D2-50, D2-100 and D3 (46, 58 and 139 s here; D1a and D1b take about 4 s each).
- **Strict xfails:** O_L for D2-50 and D2-100, and O_M3 barycentric. A future fix that makes one pass turns it
  into an XPASS failure, so the mark must then be removed and this report updated.

Without the SPICE kernels (and GL1800F for D2), the gate tests skip and name the missing file. The O_F tests
always run.

Environment of the gate run: Python 3.13.2, numpy 2.4.3, scipy 1.17.1, numba 0.65.1, spiceypy 8.2.0. During
setup, installing the wp3_1 freeze (h5py, needed by `conftest.py`) changed numpy/scipy/numba. A first gate run
had started before that change, so it was stopped and restarted. Its D1a/D1b numbers reproduced
bit-identically.

## 7. Regression

- `tools/check_repository.py`: pass (597 migrated paths, 409 preserved byte identities, 178 documents / 546
  local links).
- `tools/build_docs.py --strict`: pass (13 curated pages, including the new "External cross-validation").
- `tests/test_external_oracle.py` with `LUNAR_OD_RUN_SLOW_TESTS=1`: 19 passed, 3 xfailed (the strict xfails above).
- Full suite (`python -m pytest -p tests.desktop_preferences tests -q`): exit 0, no failures. The progress
  line counts are 2805 passed, 35 skipped and 6 xfailed; the summary line was not printed. A matplotlib
  animation callback in `examples/quickstart_od.py` prints a `ValueError` traceback without failing a test; it
  is unrelated to this work package (this branch changes no file it uses).
- No file under `src/lunar_od/` changed.
