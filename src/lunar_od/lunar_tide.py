"""Elastic solid tide of the Moon raised by the Earth and the Sun (F3 WP3.2 S1; opt-in, default-off).

Predeclaration: ``studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json``. WP3.1-R found
this term missing from production and material (230 sigma on a 1 day arc; DYN-06).

The degree-n tide raised by a body B at Moon-centred position ``d`` acts on a spacecraft at ``r``
through the potential

    V_n = k_n GM_B R^(2n+1) / (|d|^(n+1) r^(n+1)) P_n(c),    c = r_hat . d_hat,

whose gradient is ``a = A r^-(n+2) [-(n+1) P_n r_hat + P_n'(c) q]`` with ``q = d_hat - c r_hat`` and
``A = k_n GM_B R^(2n+1) / |d|^(n+1)``.  It is the IERS 2010 eq. 6.6 coefficient form summed over
orders by the addition theorem.  An elastic (no lag) tide is axially symmetric about ``d``, so no
body-fixed frame enters: the term uses the same Moon-centred Earth and Sun positions as the
third-body terms.  The Moon's deformation exerts no net force on its own centre of mass, so there is
no indirect term.

Static field convention.  The full tide is added, which is right only for a static field that
excludes the permanent tide.  GL1800F's label says so for its degree-2 and -3 coefficients.  The
GRGM660PRIM and GRGM1200L labels in this repository are silent, and the default J2 constant has no
stated tide system.  For the synchronously rotating Moon the Earth-raised tide is mostly permanent,
so pairing the full tide with a field that already contains it double counts a part comparable to
the term itself.  :class:`LunarTideOptions` therefore requires the caller to declare the convention
and accepts only ``"TIDE_FREE"``.

The functions are complex-step safe (norms are square roots of dot products and the input dtype is
kept), so their derivatives can be verified to machine precision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: GL1800F label (jggrx_1800f_sha.lbl): estimated degree-2 and -3 Love numbers and reference radius.
K2_GL1800F: float = 0.024223
K3_GL1800F: float = 0.0163
R_REF_GL1800F_M: float = 1738.0e3
GL1800F_PROVENANCE = (
    "GL1800F label jggrx_1800f_sha.lbl: k2 = 0.024223, k3 = 0.0163, reference radius 1738.0 km; "
    "'The second and third degree gravity coefficients of this model do not include the permanent tide'"
)

#: Static-field conventions this module can be paired with.
STATIC_FIELD_CONVENTIONS: tuple[str, ...] = ("TIDE_FREE",)
_UNSUPPORTED_CONVENTIONS = ("ZERO_TIDE", "MEAN_TIDE")
RAISING_BODIES: tuple[str, ...] = ("EARTH", "SUN")
SUPPORTED_DEGREES: tuple[int, ...] = (2, 3)


@dataclass(frozen=True, kw_only=True)
class LunarTideOptions:
    """Opt-in lunar solid tide.  Passing an instance enables the term; ``None`` leaves it off.

    ``static_field_convention`` has no default: the caller states that the static field used with
    this tide excludes the permanent tide (``"TIDE_FREE"``, as GL1800F's label does).  The defaults
    of the other fields are GL1800F's elastic Love numbers and reference radius.
    """

    static_field_convention: str
    love_numbers: tuple[tuple[int, float], ...] = ((2, K2_GL1800F), (3, K3_GL1800F))
    radius_m: float = R_REF_GL1800F_M
    raising_bodies: tuple[str, ...] = RAISING_BODIES
    provenance: str = GL1800F_PROVENANCE

    def __post_init__(self) -> None:
        convention = self.static_field_convention
        if not isinstance(convention, str):
            raise ValueError(
                "static_field_convention must be declared: 'TIDE_FREE' when the static field excludes "
                f"the permanent tide (GL1800F's label says so); got {convention!r}."
            )
        if convention in _UNSUPPORTED_CONVENTIONS:
            raise ValueError(
                f"static_field_convention {convention!r} is not supported: adding the tide to such a "
                "field needs its permanent part removed, and no conventional permanent tide is defined "
                "for the Moon here. Use a tide-free field (e.g. GL1800F) with 'TIDE_FREE'."
            )
        if convention not in STATIC_FIELD_CONVENTIONS:
            raise ValueError(
                f"static_field_convention must be one of {STATIC_FIELD_CONVENTIONS}; got {convention!r}."
            )

        pairs = tuple(tuple(p) for p in self.love_numbers)
        if not pairs:
            raise ValueError("love_numbers must name at least one degree.")
        degrees = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"love_numbers entries are (degree, k); got {pair!r}.")
            n, k = pair
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) not in SUPPORTED_DEGREES:
                raise ValueError(f"tide degree must be one of {SUPPORTED_DEGREES}; got {n!r}.")
            if isinstance(k, bool) or not isinstance(k, (int, float, np.integer, np.floating)):
                raise ValueError(f"Love number k{n} must be a real number; got {k!r}.")
            if not (math.isfinite(float(k)) and float(k) > 0.0):
                raise ValueError(f"Love number k{n} must be finite and > 0; got {k!r}.")
            degrees.append(int(n))
        if len(set(degrees)) != len(degrees):
            raise ValueError(f"love_numbers repeats a degree: {degrees}.")
        object.__setattr__(self, "love_numbers", tuple((int(n), float(k)) for n, k in pairs))

        radius = self.radius_m
        if isinstance(radius, bool) or not isinstance(radius, (int, float, np.integer, np.floating)) \
                or not (math.isfinite(float(radius)) and float(radius) > 0.0):
            raise ValueError(f"radius_m must be finite and > 0; got {radius!r}.")
        object.__setattr__(self, "radius_m", float(radius))

        if isinstance(self.raising_bodies, str):
            raise ValueError("raising_bodies is a tuple of body names, not a single string.")
        bodies = tuple(self.raising_bodies)
        if not bodies:
            raise ValueError("raising_bodies must name at least one body.")
        for body in bodies:
            if body not in RAISING_BODIES:
                raise ValueError(f"raising body must be one of {RAISING_BODIES}; got {body!r}.")
        if len(set(bodies)) != len(bodies):
            raise ValueError(f"raising_bodies repeats a body: {bodies}.")
        object.__setattr__(self, "raising_bodies", bodies)


def require_lunar_tide_options(value) -> "LunarTideOptions | None":
    """Return ``value`` if it is ``None`` or a :class:`LunarTideOptions`; raise ``TypeError`` otherwise."""
    if value is not None and not isinstance(value, LunarTideOptions):
        raise TypeError(f"lunar_tide must be LunarTideOptions or None; got {type(value).__name__}.")
    return value


def raising_body_gms(options: LunarTideOptions, mu_earth_m3_s2: float, mu_sun_m3_s2: float) -> tuple:
    """``(body, GM)`` pairs of the raising bodies; a GM that is not finite and > 0 raises.

    The GMs are the ones the propagator uses for the third-body terms.  A zero GM would silently
    remove that body's tide, so it is refused.
    """
    gms = {"EARTH": mu_earth_m3_s2, "SUN": mu_sun_m3_s2}
    out = []
    for body in options.raising_bodies:
        gm = gms[body]
        if not (math.isfinite(float(gm)) and float(gm) > 0.0):
            raise ValueError(
                f"the lunar tide raised by {body} needs its GM > 0; got {gm!r} (third-body GM switched "
                f"off?). Remove {body!r} from raising_bodies instead."
            )
        out.append((body, float(gm)))
    return tuple(out)


def _vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric; got dtype {arr.dtype}.")
    try:
        return arr.reshape(3)
    except ValueError:
        raise ValueError(f"{name} must hold 3 components; got shape {arr.shape}.") from None


def _legendre(n: int, c):
    """P_n(c), P_n'(c), P_n''(c) for n = 2, 3."""
    if n == 2:
        return 0.5 * (3.0 * c * c - 1.0), 3.0 * c, 3.0
    if n == 3:
        return 0.5 * (5.0 * c * c * c - 3.0 * c), 0.5 * (15.0 * c * c - 3.0), 15.0 * c
    raise ValueError(f"tide degree must be one of {SUPPORTED_DEGREES}; got {n!r}.")


def _degree_term(r, d, gm, n, k, radius, gradient: bool):
    rn = np.sqrt(np.dot(r, r))
    dn = np.sqrt(np.dot(d, d))
    rh = r / rn
    dh = d / dn
    c = np.dot(rh, dh)
    p, dp, ddp = _legendre(n, c)
    amp = k * gm * radius ** (2 * n + 1) / dn ** (n + 1)
    q = dh - c * rh
    acc = amp * rn ** -(n + 2) * (-(n + 1) * p * rh + dp * q)
    if not gradient:
        return acc, None
    rr = np.outer(rh, rh)
    grad = amp * rn ** -(n + 3) * (
        (n + 1) * (n + 2) * p * rr
        - (n + 2) * dp * (np.outer(q, rh) + np.outer(rh, q))
        + ddp * np.outer(q, q)
        - ((n + 1) * p + c * dp) * (np.eye(3) - rr)
    )
    return acc, grad


def tide_degree_acceleration(r_sc_m, r_body_m, mu_body_m3_s2, degree, love_number, radius_m) -> np.ndarray:
    """Acceleration (3,) of one degree of the tide raised by one body, m/s^2."""
    r = _vector(r_sc_m, "r_sc_m")
    d = _vector(r_body_m, "r_body_m")
    return _degree_term(r, d, mu_body_m3_s2, int(degree), love_number, radius_m, False)[0]


def tide_degree_gravity_gradient(r_sc_m, r_body_m, mu_body_m3_s2, degree, love_number, radius_m) -> np.ndarray:
    """Position gradient (3, 3) of :func:`tide_degree_acceleration`, 1/s^2.  Symmetric and traceless."""
    r = _vector(r_sc_m, "r_sc_m")
    d = _vector(r_body_m, "r_body_m")
    return _degree_term(r, d, mu_body_m3_s2, int(degree), love_number, radius_m, True)[1]


def _total(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2, options, gradient):
    if not isinstance(options, LunarTideOptions):
        raise TypeError(f"options must be LunarTideOptions; got {type(options).__name__}.")
    r = _vector(r_sc_m, "r_sc_m")
    positions = {"EARTH": _vector(r_moon_earth_m, "r_moon_earth_m"),
                 "SUN": _vector(r_moon_sun_m, "r_moon_sun_m")}
    acc = np.zeros(3, dtype=np.result_type(r, float))
    grad = np.zeros((3, 3), dtype=acc.dtype) if gradient else None
    for body, gm in raising_body_gms(options, mu_earth_m3_s2, mu_sun_m3_s2):
        for n, k in options.love_numbers:
            a_n, g_n = _degree_term(r, positions[body], gm, n, k, options.radius_m, gradient)
            acc = acc + a_n
            if gradient:
                grad = grad + g_n
    return acc, grad


def tide_acceleration(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2,
                      options: LunarTideOptions) -> np.ndarray:
    """Total tide acceleration (3,) at the Moon-centred spacecraft position, m/s^2.

    Sum over ``options.raising_bodies`` and ``options.love_numbers`` in that order.  Positions are
    Moon-centred in the propagation axes, m; GMs are the propagator's third-body GMs, m^3/s^2.
    """
    return _total(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2, options, False)[0]


def tide_gravity_gradient(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2,
                          options: LunarTideOptions) -> np.ndarray:
    """Position gradient (3, 3) of :func:`tide_acceleration`, 1/s^2."""
    return _total(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2, options, True)[1]


def tide_acceleration_and_gradient(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2,
                                   options: LunarTideOptions) -> tuple[np.ndarray, np.ndarray]:
    """``(tide_acceleration, tide_gravity_gradient)`` from one evaluation (the variational RHS)."""
    return _total(r_sc_m, r_moon_earth_m, r_moon_sun_m, mu_earth_m3_s2, mu_sun_m3_s2, options, True)
