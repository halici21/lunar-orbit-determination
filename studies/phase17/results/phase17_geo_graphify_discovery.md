# PHASE 17-GEO — Graphify discovery and manual dependency record

Graph: `graphify-out/graph.json`, built from commit `4c2fad27` (6,585 nodes). No `.py` file changed
between that build and the GEO start HEAD `fac1564`, so the graph was current and **not refreshed**.

**Graphify was used for discovery only.** Every relationship below that matters to GEO was then
read in source. A missing Graphify edge was never taken as evidence of independence.

`GRAPHIFY_DISCOVERY_GATE = PASS`: discovery was performed, then manually verified. This gate says
nothing about the physics being correct.

## Commands run

| Command | Result |
|---|---|
| `graphify explain "K_SRP"` | **Ambiguous.** It matches only two docstring (rationale) nodes; `K_SRP` is a concept, not a code symbol. Switched to real symbols. |
| `graphify affected "SRPOptions" --depth 1` | `tests/test_srp_production.py` (27), `test_srp_k_sensitivity.py` (13), `test_k_srp_bls_solve_for.py` (7), `test_k_srp_srif_solve_for.py` (4), `lunar_od/srp.py`, several phase-17 examples |
| `graphify affected "srp_acceleration_kernel"` | `lunar_od/dynamics.py` (3), `lunar_od/srp.py`, `tests/test_srp_k_sensitivity.py` |
| `graphify affected "illumination_fraction"` | `lunar_od/srp.py`, `tests/test_srp_production.py` (10), `tests/test_srp_k_sensitivity.py` |
| `graphify affected "srp_acceleration_with_lunar_shadow"` | `lunar_od/dynamics.py` (4), SRP tests |
| `graphify affected "propagate_state_with_k_sensitivity"` | `lunar_od/estimators.py` (4), `test_srp_k_sensitivity.py`, `test_r1o_stm_layout.py`, `test_k_srp_srukf_solve_for.py`, `test_k_sensitivity_fd_harness.py`, R1O-D / R1O-OPT / R1O-R examples |
| `graphify affected "_two_way_range_k_srp_column"` | `lunar_od/estimators.py`, `examples/phase17_r1m_core.py`, plus report sections in R1O, R1O-D, R1O-OPT, both literature contracts, and the R1O-R erratum |
| `graphify affected "build_range_arc"` | 8 phase-17 example scripts (R1O-D, R1O-OPT, R1O-R) |
| `graphify path "SRPOptions" "propagate_state_with_k_sensitivity"` | **No directed path.** The options object is passed as an argument, not called. |
| `graphify path "srp_acceleration_kernel" "_two_way_range_k_srp_column" --undirected` | 3 hops via `phase17_r1m_core.py`, the file that *wires* the kernel's propagator to the K column (`build_range_arc`). |
| `graphify path "campaign_initial_state" "build_range_arc" --undirected` | 1 hop (direct call) |

## Symbols discovered and then read in source

| Symbol | File | Verified content |
|---|---|---|
| `SRPOptions`, `from_cr_and_area_to_mass` | `lunar_od/srp.py` L100–207 | `K_SRP = C_R·A/m` [m²/kg]; only the product enters the runtime |
| `solar_pressure_at` | `srp.py` L213 | `P = P_1AU (AU/d)²`, true instantaneous Sun distance |
| `srp_acceleration`, `srp_acceleration_kernel` | `srp.py` L229, L358 | `a = K·P·ν·û`, û pointing Sun → spacecraft (anti-solar); kernel `g = P·ν·û` is exactly ∂a/∂K and is K-independent |
| `illumination_fraction`, `_illumination_for` | `srp.py` L307, L338 | conical penumbra, spherical Moon of radius `R_MOON_M`; exact sunward-hemisphere early exit; `NO_SHADOW` is a supported production option |
| Earth shadow | `srp.py` L415 | not modelled. Sun–Earth separation seen from the Moon at the GEO epoch is **174.4°** (near new moon), so the Earth cannot occult the Sun for any GEO orbit. |
| `_MCI_TO_MOON_BF` | `lunar_od/dynamics.py` L69–81 | J2 pole = IAU mean pole RA 269.9949°, Dec 66.5392°. GEO inclinations are defined in this frame. |
| `propagate_state_with_k_sensitivity` | `dynamics.py` L1156–1305 | 48-state history: state, Φ **column-major** `[6:42]`, `S_K` `[42:48]` |
| `two_way_range_nominal_and_initial_jacobian` | `lunar_od/two_way_range.py` L791 | re-solves the event chain on the supplied history; implicit-event Jacobian |
| `_two_way_range_k_srp_column` | `lunar_od/estimators.py` L1874 | K column via the same implicit sensitivity, with `S_K` substituted for Φ |
| `orthogonal_decomposition` | `examples/phase17_r1m_core.py` L212 | QR projector, no normal matrix |
| `square_root_covariance` | `examples/phase17_r1cov_core.py` L277 | R1COV QR route, two triangular solves |

## Dependencies found manually that Graphify does not model

These were the decisive findings of the audit, and none is visible in the graph.

1. **The fixture's force model is set by positional arguments.** `phase17_r1m_core.build_range_arc`
   calls `propagate_state_with_k_sensitivity(t, x0, C.MU, 0.0, 0.0, ...)`, so `mu_earth = mu_sun = 0`.
   The canonical baseline therefore has **no third-body gravity**. It is Moon point mass + lunar J2 +
   SRP. This is a literal value, not an edge.
2. **Epoch inconsistency inside the canonical fixture.** In the same `build_range_arc`:
   - the Sun (SRP) and Earth rotation (ITRF93) are taken at the manifest epoch
     `phase7_et0 = 857806357`;
   - the **Earth position/velocity** comes from the campaign module (`C.get_earth_pos`), whose table
     starts at `C.ET0 = 857302357`: **504,000 s = 5.83 days earlier**;
   - that table spans only 4 orbits (`C.T_OBS[-1] = 28,260 s`) and is read through `np.interp`, which
     **clamps**, so the Earth freezes after 4 orbits. At 15 orbits it is 10.65° from its true direction.

   This flows through callable injection (`get_earth_pos` is passed as a function argument) and module
   state, neither of which the graph represents.
3. **Array-slice data flow.** `S_K` is read from `nom48[:, 42:48]` and Φ from `[6:42]`. As the
   Graphify policy already records, this has no edge.

Finding (2) changes the canonical baseline materially. It is quantified in
`artifacts/phase17_geo_fixture_attribution.json` and in the main report §19.

## Likely affected

- **Modules:** GEO adds only new `examples/phase17_geo_*.py` scripts. No production module is edited.
- **Tests:** none edited. The regression set re-run is listed in the main report §53.
- **Phase reports whose absolute numbers depend on the canonical fixture:** R1M, R1COV, R1O, R1O-D,
  R1O-OPT, R1O-R. All used `build_range_arc` for the range-only baseline.

## Graphify limitations encountered this phase

- Concept names (`K_SRP`) resolve only to docstrings. Use real symbols.
- Options objects passed as arguments produce no call edge (`SRPOptions` → propagator).
- Literal argument values (`mu_earth = 0.0`) and injected callables (`get_earth_pos`) are invisible,
  and those are exactly where the fixture defects lived.
