"""Which point-mass lunar J2 model a run executes (X-02, DYN-03/04).

Standard library only, so both ``dynamics`` and the stdlib-only
``force_contract`` read the same selection.

``grail_pa_de440`` (default since the owner's 2026-11-02 decision): the J2 axis is
the MOON_PA_DE440 pole at the exact epoch, and the coefficient pairs with the
GRGM1200A reference radius (``constants.J2_MOON_GRAIL_UNNORMALIZED``,
``constants.R_MOON_GRAIL_REF_M``).

``iau_mean_pole_legacy``: the constant IAU 2006 mean pole with
(``constants.J2_MOON_UNNORMALIZED``, ``constants.R_MOON_M``), bit for bit what
every J2 baseline frozen before 2026-11-02 was computed with.

A propagator takes the model as an explicit ``lunar_j2_model`` argument.  When the
argument is omitted (every higher-level API: estimators, filters, scenarios) the
active model applies: the default, or the one selected for a block of code with
``use_lunar_j2_model``.  The selection is a context variable, so it is scoped to
the ``with`` block and to the current thread/task, never a process-wide switch.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from .constants import R_MOON_GRAIL_REF_M, R_MOON_M

LUNAR_J2_MODELS = ("grail_pa_de440", "iau_mean_pole_legacy")
DEFAULT_LUNAR_J2_MODEL = "grail_pa_de440"
LUNAR_J2_REFERENCE_RADIUS_M = {
    "grail_pa_de440": R_MOON_GRAIL_REF_M,
    "iau_mean_pole_legacy": R_MOON_M,
}

_ACTIVE: ContextVar[str] = ContextVar("lunar_j2_model", default=DEFAULT_LUNAR_J2_MODEL)


def check_lunar_j2_model(model: str) -> str:
    if model not in LUNAR_J2_MODELS:
        raise ValueError(f"lunar_j2_model must be one of {LUNAR_J2_MODELS}, got {model!r}")
    return model


def active_lunar_j2_model() -> str:
    """The model an omitted ``lunar_j2_model`` argument resolves to."""
    return _ACTIVE.get()


def resolve_lunar_j2_model(model: str | None) -> str:
    return check_lunar_j2_model(active_lunar_j2_model() if model is None else model)


@contextmanager
def use_lunar_j2_model(model: str):
    """Run a block with ``model`` as the active lunar J2 model."""
    token = _ACTIVE.set(check_lunar_j2_model(model))
    try:
        yield model
    finally:
        _ACTIVE.reset(token)
