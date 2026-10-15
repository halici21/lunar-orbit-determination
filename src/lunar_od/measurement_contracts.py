"""Physical input contracts for the measurement generators (Phase 17-HARDEN).

``lunar_od/two_way_range.py`` is byte-frozen by the qualified R3-P21-M3 gate
(``tests/test_od_contracts.py``), so its input contract is enforced here, one
layer up, rather than inside it.  Every production and Phase-17 caller
generates two-way range through :func:`generate_two_way_range_measurements_checked`;
a repository test forbids direct calls from the Phase-17 science code.

The contract, per call:

* an epoch-tagged Earth provider must share ``et0``'s ET origin and be
  J2000 / Moon-centred / metres (:func:`lunar_od.ephemeris.require_ephemeris_contract`);
  a provider marked ``legacy_reproduction_only`` is accepted with a warning;
* any provider must give a physically plausible Moon-Earth distance over the
  pass (catches km-for-m, wrong centre and non-physical stand-ins).

Phase 17-FXR's defect (an Earth table whose seconds counted from an ET origin
504000 s earlier than ``et0``) passed the unguarded generator silently.
"""
from __future__ import annotations

import numpy as np

from .ephemeris import require_ephemeris_contract, require_moon_earth_distance
from .two_way_range import generate_two_way_range_measurements


def require_earth_provider_contract(get_earth_pos, get_earth_vel, et0: float, t_s) -> tuple[str, str]:
    """Validate an Earth position/velocity provider pair for a pass at ``t_s``."""
    status_pos = require_ephemeris_contract(get_earth_pos, et0, "get_earth_pos")
    status_vel = require_ephemeris_contract(get_earth_vel, et0, "get_earth_vel")
    t = np.asarray(t_s, dtype=float).reshape(-1)
    require_moon_earth_distance(np.asarray(get_earth_pos(t), dtype=float).reshape(t.size, 3),
                                "get_earth_pos(t_pass_s)")
    return status_pos, status_vel


def generate_two_way_range_measurements_checked(
    t_pass_s, state_history_mci, stations, vis_mask_raw, get_earth_pos, get_earth_vel,
    et0: float, **kwargs,
):
    """:func:`lunar_od.two_way_range.generate_two_way_range_measurements` behind the
    Earth-provider contract.  Arguments and return value are identical."""
    require_earth_provider_contract(get_earth_pos, get_earth_vel, et0, t_pass_s)
    return generate_two_way_range_measurements(
        t_pass_s, state_history_mci, stations, vis_mask_raw, get_earth_pos, get_earth_vel,
        et0, **kwargs)
