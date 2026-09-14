"""Phase 17A-R - permanent guards for the K_SRP finite-difference harness.

These tests protect the FD *harness*, not the physics. Phase 17A lost a whole
campaign to two harness defects that produced plausible-looking numbers:

  P17A-D1  the perturbed state history was built correctly, but the observable
           was then requested from the NOMINAL history. Every finite difference
           came back exactly zero and was faithfully reported as a relative
           error of 1.000 at every step. It was caught only because Layer A and
           Layer B agreed to four significant figures, which is impossible.

  P17A-D2  the counted-Doppler light-time tolerance sat above the derivative
           signal, so the solver's own convergence -- not the physics -- set
           the finite difference.

Both are silent failures: the code runs, the numbers look like numbers. So they
are pinned here rather than left to manual inspection.

The guards assert *directionality and responsiveness*, never a value, so they
stay valid as the models evolve.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spiceypy")

from lunar_od.config import position_only_stations
from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2
from lunar_od.dynamics import propagate_state_with_k_sensitivity
from lunar_od.radiometrics import (
    RangeRatePhysicsConfig,
    two_way_counted_doppler_observable,
)
from lunar_od.spice_loader import load_spice_kernels
from lunar_od.srp import SRPOptions
from lunar_od.two_way_range import (
    TwoWayRangeConfig,
    make_exact_sxform_station_state_provider,
    solve_two_way_range_events,
    two_way_range_from_solution,
)

K_NOMINAL = 0.01
CADENCE = 60.0
TIGHT = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14,
                          max_iter=200)


@pytest.fixture(scope="module")
def arc():
    """A short illuminated arc carrying a real dx/dK sensitivity."""
    import spiceypy as spice

    from lunar_od.visibility import sample_j2000_to_itrf93_transforms

    load_spice_kernels(None, clear=True)
    et0 = 857806356.99998
    t_grid = np.arange(0.0, 12000.0, CADENCE)
    x0 = np.array([-1746513.5335792408, -277083.98513177276, 518734.13974716066,
                   -138.53981552770264, -1206.7373147449678, -1085.0511650068934])

    def earth_at(_t):
        return np.array([384_400e3, 0.0, 0.0])

    def sun_at(t):
        return spice.spkezr("SUN", et0 + float(t), "J2000", "NONE",
                            "MOON")[0][:3] * 1000.0

    nom = propagate_state_with_k_sensitivity(
        t_grid, x0, MU_MOON_M3S2, 0.0, 0.0, earth_at, sun_at,
        srp=SRPOptions(k_srp_m2_per_kg=K_NOMINAL), rtol=1e-13, atol=1e-16,
        j2_moon=J2_MOON_UNNORMALIZED)
    states, s_k = nom[:, :6], nom[:, 42:48]

    n = t_grid.size
    epos = np.tile(np.array([384_400e3, 0.0, 0.0]), (n, 1))
    evel = np.zeros((n, 3))
    station = [s for s in position_only_stations() if "Goldstone" in s.name][0]
    provider = make_exact_sxform_station_state_provider(
        station, float(et0), t_grid, epos, evel)
    xf = sample_j2000_to_itrf93_transforms(float(et0), t_grid)
    return dict(et0=et0, t_grid=t_grid, states=states, s_k=s_k,
                provider=provider, station=station, epos=epos, evel=evel,
                xf=xf)


def _range(arc, t_rx, history):
    return two_way_range_from_solution(
        solve_two_way_range_events(t_rx, arc["provider"], arc["t_grid"],
                                   history, TIGHT), TIGHT)


def _doppler(arc, t_mid, history, tol=1e-13):
    cfg = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler", count_interval_s=CADENCE,
        counted_doppler_model="four_event_delay", output_unit="mps_equivalent",
        light_time_tolerance_s=tol, light_time_equation_tolerance_s=0.1 * tol)
    return float(two_way_counted_doppler_observable(
        t_mid, arc["station"], arc["t_grid"], history, arc["epos"],
        arc["evel"], arc["xf"], cfg, et0_s=arc["et0"]))


T_PROBE = 9600.0


# ======================================================================
# P17A-D1 - the perturbed history must reach the observable
# ======================================================================
def test_range_observable_responds_to_a_perturbed_history(arc):
    """A state history perturbed along S_K must move the range.

    This is the exact defect that voided the first Phase 17A campaign: the
    perturbation was built but never reached the evaluated observable.
    """
    dk = 1e-2 * K_NOMINAL
    base = _range(arc, T_PROBE, arc["states"])
    plus = _range(arc, T_PROBE, arc["states"] + dk * arc["s_k"])
    minus = _range(arc, T_PROBE, arc["states"] - dk * arc["s_k"])
    assert plus != base and minus != base, (
        "the two-way range did not move for a perturbed state history; the "
        "harness is evaluating the nominal trajectory (P17A-D1)"
    )
    assert plus != minus


def test_counted_doppler_responds_to_a_perturbed_history(arc):
    """Same guard on the four-event counted-Doppler composition path."""
    dk = 1e-2 * K_NOMINAL
    base = _doppler(arc, T_PROBE, arc["states"])
    plus = _doppler(arc, T_PROBE, arc["states"] + dk * arc["s_k"])
    minus = _doppler(arc, T_PROBE, arc["states"] - dk * arc["s_k"])
    assert plus != base and minus != base, (
        "counted Doppler did not move for a perturbed state history (P17A-D1)"
    )
    assert plus != minus


@pytest.mark.parametrize("observable", ["range", "doppler"])
def test_the_central_difference_is_not_identically_zero(arc, observable):
    """The failure mode reported a relative error of exactly 1.000 everywhere."""
    dk = 1e-2 * K_NOMINAL
    ev = _range if observable == "range" else _doppler
    num = (ev(arc, T_PROBE, arc["states"] + dk * arc["s_k"])
           - ev(arc, T_PROBE, arc["states"] - dk * arc["s_k"]))
    assert num != 0.0
    assert np.isfinite(num)


def test_the_response_scales_with_the_perturbation(arc):
    """Halving dK must roughly halve the numerator.

    A harness that silently ignores its input cannot do this, and neither can
    one whose observable is quantised above the signal.
    """
    base = _range(arc, T_PROBE, arc["states"])
    big = 1e-2 * K_NOMINAL
    n_big = _range(arc, T_PROBE, arc["states"] + big * arc["s_k"]) - base
    n_half = _range(arc, T_PROBE, arc["states"] + 0.5 * big * arc["s_k"]) - base
    assert n_big != 0.0 and n_half != 0.0
    assert n_half / n_big == pytest.approx(0.5, rel=0.05)


# ======================================================================
# P17A-D2 - the solver tolerance must sit below the signal
# ======================================================================
def test_counted_doppler_fd_is_invariant_to_a_tighter_solver(arc):
    """Tightening the light-time solver 10x must not move the derivative.

    If it does, the solver's convergence -- not the measurement physics -- is
    setting the finite difference, which is what P17A-D2 was.
    """
    dk = 1e-2 * K_NOMINAL
    hp, hm = arc["states"] + dk * arc["s_k"], arc["states"] - dk * arc["s_k"]
    loose = (_doppler(arc, T_PROBE, hp, 1e-13)
             - _doppler(arc, T_PROBE, hm, 1e-13)) / (2.0 * dk)
    tight = (_doppler(arc, T_PROBE, hp, 1e-14)
             - _doppler(arc, T_PROBE, hm, 1e-14)) / (2.0 * dk)
    assert loose != 0.0
    assert abs(tight - loose) / abs(loose) < 1e-6, (
        "the counted-Doppler derivative moved when the light-time solver was "
        "tightened; the solver tolerance is the limiting layer (P17A-D2)"
    )


def test_the_default_doppler_tolerance_is_documented_as_too_loose(arc):
    """The 1e-10 s default is ~3 cm of range, ~1e-3 m/s over a 60 s count.

    Pinned so nobody quietly runs a K-sensitivity campaign at the default and
    rediscovers P17A-D2 the hard way.
    """
    default = RangeRatePhysicsConfig(mode="two_way_counted_doppler")
    equivalent_mps = (2.0 * 299792458.0 * default.light_time_tolerance_s
                      / CADENCE)
    assert default.light_time_tolerance_s == 1e-10
    assert equivalent_mps > 1e-4, (
        "the default tolerance is no longer the hazard this test describes; "
        "update the guard rather than deleting it"
    )


# ======================================================================
# The FD oracle must stay independent of the analytic derivative
# ======================================================================
def test_the_fd_path_does_not_import_the_jacobian(arc):
    """The oracle evaluates the observable; it must never call the Jacobian.

    A finite difference that reuses analytic derivative code validates nothing.
    """
    import inspect

    from lunar_od import two_way_range as twr

    src = inspect.getsource(twr.two_way_range_from_solution)
    assert "jacobian" not in src.lower()
    assert "sensitivity" not in src.lower()
