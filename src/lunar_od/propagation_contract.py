"""Explicit transport of force options and the effective numerical profile.

No-options calls preserve their historical integration path. An explicit
profile, or enabled SRP, selects one fixed profile for all solve consumers.
"""
from dataclasses import asdict
from .dynamics import PropagationProfile
from .srp import SRPOptions


def resolve_profile(srp=None, propagation_profile=None):
    if srp is not None and not isinstance(srp, SRPOptions):
        raise TypeError('srp must be SRPOptions or None.')
    if propagation_profile is not None and not isinstance(propagation_profile, PropagationProfile):
        raise TypeError('propagation_profile must be PropagationProfile or None.')
    profile = propagation_profile
    if profile is None and srp is not None and srp.enabled:
        profile = PropagationProfile()
    if profile is not None:
        if profile.method not in ('ADAMS','DOP853','RK45','RK23','Radau','BDF','LSODA'):
            raise ValueError(f'Unsupported propagation profile method: {profile.method!r}.')
        if type(profile.shadow_events) is not bool:
            raise ValueError('propagation_profile.shadow_events must be boolean.')
        if profile.shadow_events and profile.method != 'DOP853':
            raise ValueError('A shadow-event profile requires DOP853.')
    return profile


def model_options(srp=None, propagation_profile=None):
    """Options passed between consumers, keeping legacy calls unchanged."""
    profile = resolve_profile(srp, propagation_profile)
    out = {} if srp is None else {'srp': srp}
    if profile is not None:
        out['propagation_profile'] = profile
    return out


def propagator_options(srp=None, propagation_profile=None):
    options = model_options(srp, propagation_profile)
    profile = options.pop('propagation_profile', None)
    if profile is not None:
        options.update(profile.propagator_kwargs())
    return options


def numerical_metadata(srp, propagation_profile, rtol, atol, *, fast_sigma=False,
                       local_state_model='ode', adaptive_tolerances=True):
    profile = resolve_profile(srp, propagation_profile)
    shadowed = srp is not None and srp.enabled and srp.shadow_model == 'CONICAL_PENUMBRA'
    effective_local = 'ode' if shadowed and local_state_model == 'taylor3' else local_state_model
    return dict(schema_version='numerical-profile.v1',
        method='RK4_FAST_SIGMA' if fast_sigma else profile.method if profile else 'ADAMS',
        shadow_events=bool(profile and profile.shadow_events),rtol=float(rtol),atol=float(atol),
        error_control='fixed_step_external_sigma' if fast_sigma else
            'legacy_adaptive_estimator_tolerances' if profile is None and adaptive_tolerances
            else 'fixed_caller_tolerances',
        rtol_atol_apply_to_sigma=not fast_sigma,
        requested_local_state_model=local_state_model,effective_local_state_model=effective_local,
        srp_runtime=None if srp is None else asdict(srp))


def truth_numerical_metadata(srp, propagation_profile, rtol, atol):
    """The shared truth wrapper historically defaults to DOP853, not ADAMS."""
    profile=resolve_profile(srp,propagation_profile)
    if profile is None:
        profile=PropagationProfile(method='DOP853',shadow_events=False)
    return numerical_metadata(srp,profile,rtol,atol,adaptive_tolerances=False,
                              local_state_model='not_applicable')
