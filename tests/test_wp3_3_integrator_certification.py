"""F3 WP3.3 long-arc integrator certification: gate K_G1 and the C_G1 validity rule (REQ-DYN-006).

Predeclaration: studies/phase17/configs/wp3_3_integrator_certification_predeclaration.json.
Report: studies/phase17/reports/wp3_3_integrator_certification.md.

K_G1 (test_two_body_10rev_closure) needs no SPICE: pure Kepler about the Moon, Earth and Sun GMs zero.
Default tier: P_ADAMS, P_DOP853, P_D2; the other profiles run with LUNAR_OD_RUN_SLOW_TESTS=1 (and
always in the driver).
"""
from __future__ import annotations

import os

import pytest

import wp3_3_integrator_certification as C
from lunar_od.dynamics import PropagationProfile
from lunar_od.propagation_contract import resolve_profile

RUN_SLOW = os.environ.get("LUNAR_OD_RUN_SLOW_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}
FAST_PROFILES = ("P_ADAMS", "P_DOP853", "P_D2")

#: K_G1 cases that failed when first run (report section 2): strict xfail, no tolerance change.
K_G1_RECORDED_FAILURES: dict[tuple[str, str], str] = {}


def _k_g1_params():
    for profile in C.PROFILES:
        for case in C.KEPLER_CASES:
            marks = []
            if profile not in FAST_PROFILES:
                marks.append(pytest.mark.skipif(not RUN_SLOW, reason="slow profile: LUNAR_OD_RUN_SLOW_TESTS=1"))
            reason = K_G1_RECORDED_FAILURES.get((profile, case))
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            yield pytest.param(profile, case, marks=marks, id=f"{profile}-{case}")


@pytest.mark.parametrize(("profile", "case"), list(_k_g1_params()))
def test_two_body_10rev_closure(profile, case):
    """K_G1: after 10 Keplerian periods the state returns to r0 within 1 mm, energy within 1e-11 relative."""
    out = C.two_body_closure(profile, case)
    assert out["revolutions"] == 10
    assert out["closure_m"] <= C.K_G1_CLOSURE_M, out
    assert out["energy_rel_max"] <= C.K_G1_ENERGY_REL, out


def test_profile_set_is_every_legal_profile():
    """The certified set is resolve_profile's whitelist with shadow_events False, plus the DOP853 event profile."""
    assert set(C.PROFILES) == {"P_ADAMS", "P_DOP853", "P_D2", "P_RK45", "P_RK23", "P_RADAU", "P_BDF", "P_LSODA"}
    methods = {p.method for p in C.PROFILES.values()}
    for name, profile in C.PROFILES.items():
        assert resolve_profile(None, profile) is profile, name
    assert C.PROFILES["P_D2"] == PropagationProfile()
    for method in methods - {"DOP853"}:
        with pytest.raises(ValueError):
            resolve_profile(None, PropagationProfile(method=method, shadow_events=True))
    with pytest.raises(ValueError):
        resolve_profile(None, PropagationProfile(method="RK4", shadow_events=False))


def _cell(errors, status=None):
    """Synthetic cell: {arc_days: (e_r, status)}."""
    status = status or {}
    return {str(d): {"status": status.get(d, "OK"), "e_r_m": e} for d, e in errors.items()}


def test_validity_rule_takes_longest_arc_passing_on_every_orbit():
    ref = {o: {str(d): 1e-6 for d in C.ARCS_DAYS} for o in ("A", "B")}
    cells = {"A": _cell({1: 1e-4, 3: 5e-4, 7: 2e-3, 28: 5e-2}),
             "B": _cell({1: 1e-4, 3: 2e-3, 7: 3e-3, 28: 4e-3})}
    out = C.validity_range(cells, ref)
    assert out == {"MM": 1, "CM": 7, "DM": 28, "M": 28}


def test_validity_rule_none_budget_and_inconclusive():
    ref = {"A": {"1": 1e-6, "3": 1e-6, "7": 5e-4, "28": 5e-4}}
    # MM at 7 d: reference self-difference 5e-4 > 0.1 x 1e-3 -> INCONCLUSIVE, so MM stops at 3 d
    out = C.validity_range({"A": _cell({1: 1e-5, 3: 1e-5, 7: 1e-5, 28: 1e-5})}, ref)
    assert out["MM"] == 3 and out["CM"] == 28
    # budget at 7 d stops every tier at 3 d; an error above 1 m at 1 d gives 'none'
    out = C.validity_range({"A": _cell({1: 2.0, 3: 2.0, 7: None, 28: None},
                                       {7: "BUDGET_EXCEEDED", 28: "NOT_ATTEMPTED"})}, ref)
    assert out == {"MM": "none", "CM": "none", "DM": "none", "M": "none"}
    out = C.validity_range({"A": _cell({1: 1e-5, 3: 1e-5, 7: None, 28: None},
                                       {7: "BUDGET_EXCEEDED", 28: "NOT_ATTEMPTED"})}, ref)
    assert out == {"MM": 3, "CM": 3, "DM": 3, "M": 3}
    out = C.validity_range({"A": _cell({1: None, 3: None, 7: None, 28: None},
                                       {1: "ERROR", 3: "NOT_ATTEMPTED", 7: "NOT_ATTEMPTED",
                                        28: "NOT_ATTEMPTED"})}, ref)
    assert set(out.values()) == {"none"}
