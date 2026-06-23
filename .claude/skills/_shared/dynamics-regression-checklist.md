# Dynamics Regression Checklist

Use when modifying force models or propagation (`lunar_od/dynamics.py`,
`lunar_od/accelerated.py`).

Verify:
- acceleration consistency
- propagation agreement (vs prior behavior / registered baseline)
- STM consistency (finite-difference check)
- frame transformation consistency
- ephemeris consistency
- third-body (Earth / Sun) consistency
- J2 / Jn consistency
- integration tolerance sensitivity
- fast path (accelerated) vs slow path agreement

Targeted tests: `test_dynamics.py` (and `test_ephemeris.py`,
`test_spice_loader.py` if frames / SPICE are touched). See `test-impact-map.md`.
