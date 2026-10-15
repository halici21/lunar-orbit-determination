# Phase 17-FXR — Graphify refresh and dependency discovery

Graphify is a discovery aid. The source code is authoritative. Every edge used
below was confirmed by opening the file it points to.

## Refresh

| field | value |
|---|---|
| `GRAPHIFY_VERSION` | graphify 0.9.65 |
| `GRAPHIFY_REFRESHED` | YES — `graphify update .` (code-only, no LLM, ~12 s) |
| `GRAPHIFY_REFRESH_HEAD` | `40bc1533fb3056b626b3132bbd71a153456c16da` (entering HEAD, before any FXR edit) |
| result | 6795 nodes, 18476 edges, 223 communities; 14 previously uncached files re-extracted, which included the GEO scripts the hygiene audit reported as missing |

The graph is per-worktree and gitignored. It was refreshed again after the FXR code
changes, so that the new FXR modules are included.

## Queries run

| query | result | used for |
|---|---|---|
| `explain "build_range_arc"` | `examples/phase17_r1m_core.py` L89, degree 71; callees `campaign_epoch`, `propagate_state_with_k_sensitivity`, `sample_j2000_to_itrf93_transforms`, `analyze_visibility_gap_with_transforms`, `generate_two_way_range_measurements`, `two_way_range_nominal_and_initial_jacobian`, `_two_way_range_k_srp_column` | locating the defect site |
| `affected "build_range_arc" --depth 2` | 25 caller scripts: R1COV (abc, d_srif, g_requalify, h_artifacts), R1M (addendum, config_and_spectra, covariance_diagnostic, multi_arc_study), R1O (baseline, celestial, comparison, confirmation, crosscheck, ddor, landmark), R1O-OPT (bridge, campaign, jacobian_and_k), R1O-R (requalification), R1O-D (anchor_convention_check, campaign, jacobian_and_k, surrogate_bridge), GEO (campaign, fixture_attribution) | blast radius of the fixture |
| `path "build_range_arc" "_two_way_range_k_srp_column" --undirected` | 1 hop (`calls`) | the fixture wires the K column |
| `path "build_range_arc" "generate_two_way_range_measurements" --undirected` | 1 hop (`calls`) | the fixture hands Earth to production range |
| `affected "_two_way_range_k_srp_column" --depth 2` | production `estimate_two_way_range_bls_lm`, `estimate_two_way_range_srif`, `_two_way_range_posterior_design_with_k`; analysis `build_range_arc`, `build_geo_range_arc` | confirms production consumers are the estimator K paths (not touched by FXR) |
| `explain "make_exact_sxform_station_state_provider"` | production `lunar_od/two_way_range.py`; calls `_interp_state` (Earth history Hermite interpolation), `spice.sxform` at `et0 + t` | where the Earth callable's values enter the event solver |
| `explain "square_root_covariance"` | `examples/phase17_r1cov_core.py` L277 | R1COV path used by every metric |
| `affected "build_ddor_arc" --depth 1` | R1O comparison, confirmation, crosscheck, ddor; R1O-R requalification; R1O-D surrogate bridge | second Earth consumer (surrogate ΔDOR) |
| `explain "campaign_epoch"` | degree 58; imported by every Phase 17 R1x/GEO script | the scenario-epoch source |
| `explain "_earth_fns"` | GEO's diagnostic Earth providers | reused as the attribution reference |
| `explain "get_earth_pos"` | **ambiguous: five `get_earth_pos` nodes, all test-local helpers** (`test_k_srp_*`, `test_two_way_range_integration`, `test_estimator_lm_consistency`). The campaign module's `get_earth_pos` is **not a node at all** | see below |

## What the graph could not show

The defect is invisible to static dependency analysis, for three independent reasons:

1. **The Earth table lives outside the graphed tree.** `od_gravity_covariance_campaign.py`
   is in the separate `od_covariance_campaign` repository. It is imported at runtime
   (`import od_gravity_covariance_campaign as C` inside `build_range_arc`) and is found
   through `PYTHONPATH`. It has no node, so `C.get_earth_pos` has no edge.
2. **The Earth is passed as a runtime callable argument.** Production
   `generate_two_way_range_measurements(..., get_earth_pos, get_earth_vel, et0, ...)`
   receives an untagged function of scenario seconds. What epoch that function's
   `t = 0` means cannot be expressed as an edge. The mismatch was between two numbers:
   the `et0` argument (`phase7_et0`) and the table's hidden origin (`C.ET0`, 504000 s
   earlier).
3. **The clamp is a property of `numpy.interp`, not of the call graph.**

The complete list of direct `C.get_earth_pos` / `C.get_earth_vel` consumers came
from `Grep`, not from the graph:

| file | site | Earth use |
|---|---|---|
| `phase17_r1m_core.py` | `build_range_arc` | visibility + production range (the fixture) |
| `phase17_r1o_core.py` | `build_ddor_arc` | surrogate ΔDOR station placement, moving |
| `phase17_r1o_r_requalification.py` | `earth_los_rows`, `bridge_reduced_rows` | LOS control (moving), bridge (frozen at t = 0) |
| `phase17_r1od_campaign.py` | `build_ddor_rows`, `main` | production ΔDOR station placement, **frozen at t = 0 for the whole arc** |
| `phase17_r1o_baseline.py` | window probe | elevation probe |
| `phase17_r1o_celestial.py` | Earth-LOS rows | LOS control |
| `phase17_r1od_anchor_convention_check.py`, `phase17_r1od_jacobian_and_k.py`, `phase17_r1od_surrogate_bridge.py` | frozen `get_earth_pos(0.0)` | software qualification geometry |
| `phase17_r1od_fd_convergence.py`, `phase17_r1od_qualification.py` | (untracked R1O-D orphans) | **not modified** (FXR s48) |

`GRAPHIFY_DISCOVERY_GATE = PASS`: the queries located every tracked consumer of the
fixture. The runtime-argument channel that carried the defect was covered by the
manual audit (`FXR_RUNTIME_EPOCH_AUDIT`) and by Grep.

Raw query output: session scratch (`graphify_queries.txt`), not committed.
