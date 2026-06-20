# Performance Optimization Plan

Scope: computational speed improvements only. Mathematical results must be bit-identical (or within floating-point rounding) before and after each task. Run `python -m pytest python_port/tests/ -q` after every task — must stay at 172 passed.

Excluded from this plan: light-time correction (correctness), rank-revealing QR (robustness), Q auto-calibration (functionality), angle-wrap Jacobian (correctness).

---

## TASK-P1 — Analytical J₂ Gravity-Gradient Tensor

**File:** `python_port/lunar_od/dynamics.py`  
**Function:** `zonal_j2_gravity_gradient(r_bf, mu, radius, j2)`  
**Current cost:** 6 extra `zonal_j2_acceleration(...)` calls per Jacobian evaluation (finite-difference).  
**Target:** Replace with a single closed-form tensor evaluation.

### Closed-form expression (body-fixed frame, z = pole axis)

```
r   = |r_bf|
z   = r_bf[2]
u   = z / r
ρ   = sqrt(r_bf[0]² + r_bf[1]²)   # equatorial distance

Prefactor:
  α  = (3/2) * J2 * (R/r)² * (μ/r³)

J2 gravity-gradient tensor G_j2 (3×3 symmetric):

  G_j2[i,j] = α * [
      (5u² - 1) * δ[i,j]
    - 5 * (5u² - 3) * r_bf[i]*r_bf[j] / r²
    + 5 * (r_bf[i]*e_z[j] + r_bf[j]*e_z[i]) * u
    - 35 * u² * r_bf[i]*r_bf[j] / r²          ← only for off-diagonal z-containing terms
  ]

Expanded per element (e_z = [0,0,1]):

  G_j2[0,0] = α * (5u² - 1  - 5*(5u²-3)*x²/r²)
  G_j2[1,1] = α * (5u² - 1  - 5*(5u²-3)*y²/r²)
  G_j2[2,2] = α * (5u² - 1  - 5*(5u²-3)*z²/r²  + 10*u²)   ← extra +10u² from e_z·e_z
  G_j2[0,1] = α * (           - 5*(5u²-3)*x*y/r²)
  G_j2[0,2] = α * (5*u*x/r²  - 5*(5u²-3)*x*z/r²)
  G_j2[1,2] = α * (5*u*y/r²  - 5*(5u²-3)*y*z/r²)
  (symmetric: G_j2[j,i] = G_j2[i,j])
```

Total: `G = G_pm + G_j2` where `G_pm[i,j] = (μ/r³)(3*r_bf[i]*r_bf[j]/r² - δ[i,j])`.

**Verification:** Existing finite-difference test must agree to < 1e-6 relative. The function signature and return shape do not change.

**Estimated gain:** Eliminates 6 force evaluations per STM Jacobian call. Most impactful at high ODE tolerances (many internal steps) or long arcs.

---

## TASK-P2 — Numba Fast Path for J₂

**File:** `python_port/lunar_od/dynamics.py` (and `accelerated.py` if Numba kernels live there)  
**Current behavior:** `use_fast = _FAST_DYNAMICS and not j2_moon` — J₂ disables Numba entirely.  
**Target:** Inline J₂ into the JIT kernel; restore ~10× ODE RHS speedup for J₂-enabled scenarios.

### Steps

1. Read the current `_f3body_rhs_fast` Numba kernel signature and body in `accelerated.py` (or wherever it lives).

2. Add parameters `j2_moon: float`, `moon_r: float`, `c_bf: float[:,:]` (the 3×3 `_MCI_TO_MOON_BF` matrix as a `numba.typed` or plain numpy array passed at call time).

3. Append to the kernel body — after the existing point-mass + third-body block:

```python
if j2_moon != 0.0:
    # rotate to body-fixed
    rbx = c_bf[0,0]*rx + c_bf[0,1]*ry + c_bf[0,2]*rz
    rby = c_bf[1,0]*rx + c_bf[1,1]*ry + c_bf[1,2]*rz
    rbz = c_bf[2,0]*rx + c_bf[2,1]*ry + c_bf[2,2]*rz
    rb  = math.sqrt(rbx*rbx + rby*rby + rbz*rbz)
    u   = rbz / rb
    fac = 1.5 * j2_moon * (moon_r/rb)**2 * mu_moon / rb**3
    # J2 acceleration in body-fixed frame
    abx = fac * rbx * (1.0 - 5.0*u*u)
    aby = fac * rby * (1.0 - 5.0*u*u)
    abz = fac * rbz * (3.0 - 5.0*u*u)
    # rotate back to MCI  (C.T @ a_bf  =  c_bf transposed)
    ax += c_bf[0,0]*abx + c_bf[1,0]*aby + c_bf[2,0]*abz
    ay += c_bf[0,1]*abx + c_bf[1,1]*aby + c_bf[2,1]*abz
    az += c_bf[0,2]*abx + c_bf[1,2]*aby + c_bf[2,2]*abz
```

4. In `f3body_moon(...)` (Python wrapper): always pass `_MCI_TO_MOON_BF`, `MOON_R_M`, `j2_moon` to the kernel (pass zeros/0.0 when J₂ off — the `if j2_moon != 0.0` guard is free).

5. Remove the `not j2_moon` gate: `use_fast = _FAST_DYNAMICS`.

**Constraint:** No Python function calls inside the JIT kernel — use `math.sqrt`, scalar arithmetic, direct array indexing only.

**Verification:**
```bash
python -c "
import numpy as np
from lunar_od.dynamics import f3body_moon
r = np.array([0., 0., 1937400., 0., 0., 0.])
a_py  = f3body_moon(r, 4.9048695e12, 0, 0, np.zeros(3), np.zeros(3), j2_moon=2.0346e-4)[3:]
print('J2 accel z:', a_py[2])
"
```
Result must match the pre-TASK value to 1e-10 relative.

**Estimated gain:** 5–10× ODE RHS throughput for J₂-enabled scenarios (matches J₂-off Numba baseline).

---

## TASK-P3 — Pre-Computed STM Cache for BLS/SRIF Iterations

**File:** `python_port/lunar_od/estimators.py`  
**Functions:** `estimate_position_bls_lm`, `estimate_range_rate_bls_lm`, `estimate_position_srif`, `estimate_range_rate_srif`  
**Current cost:** Full 42-state ODE integration on **every** BLS/SRIF iteration.  
**Target:** Integrate the augmented state once (first iteration); reuse the STM array for all subsequent iterations.

### Change pattern (apply identically to all 4 functions)

Add `use_stm_cache: bool = True` keyword argument.

Inside the iteration loop, replace:
```python
# current: runs every iteration
x_aug = propagate_augmented_state(t_grid, x_aug0, ..., rtol=rtol, atol=atol, j2_moon=j2_moon)
x_history = x_aug[:, :6]
stm_history = x_aug[:, 6:].reshape(-1, 6, 6, order='F')   # (N, 6, 6)
```

With:
```python
if use_stm_cache and _stm_cache is not None and _stm_invalid is False:
    # cheap: 6-state ODE only
    x_history = propagate_state(t_grid, x0, ..., rtol=rtol, atol=atol, j2_moon=j2_moon)
    stm_history = _stm_cache
else:
    # full augmented propagation
    x_aug = propagate_augmented_state(t_grid, x_aug0, ..., rtol=rtol, atol=atol, j2_moon=j2_moon)
    x_history  = x_aug[:, :6]
    stm_history = x_aug[:, 6:].reshape(-1, 6, 6, order='F')
    _stm_cache  = stm_history
    _stm_invalid = False
```

After computing `delta_x` each iteration:
```python
if np.linalg.norm(delta_x[:3]) > 1e3:   # > 1 km step → re-propagate STM next iter
    _stm_invalid = True
```

Initialize `_stm_cache = None`, `_stm_invalid = False` before the loop.

**Verification:** With `use_stm_cache=True` and `use_stm_cache=False`, `EstimatorStats.final_cost` must agree to 1e-6 relative on all existing estimator tests.

**Estimated gain:** N_iter − 1 fewer 42-state ODE integrations per arc. For 10 iterations: ~9× reduction in augmented propagation calls after the first iteration.

---

## TASK-P4 — Vectorized Measurement Jacobian Assembly

**File:** `python_port/lunar_od/measurements.py`  
**Functions:** `compute_position_residuals_analytic`, `compute_range_rate_residuals_analytic`  
**Current cost:** Python loop over N epochs to build H matrix and residual vector.  
**Target:** Batch numpy operations — no Python epoch loop.

### Position Jacobian (vectorized)

```python
# Inputs:
#   state_history: (N, 6)   MCI state at each epoch
#   station_ecef:  (3,)     station ECEF position
#   C_sez:         (3, 3)   = ecef2sez_dcm(lat, lon)  — computed once, not per epoch

r_rel   = state_history[:, :3] - station_ecef          # (N, 3) MCI→station delta
# Note: need MCI→ECEF rotation per epoch if frames differ; if same frame, direct subtraction.
# Check existing code for whether state_history is already in ECEF at this point.

rng     = np.linalg.norm(r_rel, axis=1)                # (N,)
r_hat   = r_rel / rng[:, None]                         # (N, 3)

# SEZ components (batch)
r_sez   = (C_sez @ r_rel.T).T                          # (N, 3)  [S, E, Z]
rho_xy  = np.hypot(r_sez[:, 0], r_sez[:, 1])          # (N,)  horizontal range

# ∂range/∂r  (3,) per epoch → (N, 3)
dH_rng  = np.hstack([r_hat, np.zeros((N, 3))])         # (N, 6)

# ∂azimuth/∂r = (1/rho_xy²) [-E, S, 0] mapped back to MCI
daz_sez = np.column_stack([-r_sez[:,1], r_sez[:,0], np.zeros(N)]) / rho_xy[:, None]**2
dH_az   = np.hstack([(C_sez.T @ daz_sez.T).T, np.zeros((N, 3))])  # (N, 6)

# ∂elevation/∂r = [-S*Z/r*rho_xy, -E*Z/r*rho_xy, rho_xy/r²] mapped to MCI
del_sez = np.column_stack([
    -r_sez[:,0]*r_sez[:,2] / (rng * rho_xy),
    -r_sez[:,1]*r_sez[:,2] / (rng * rho_xy),
     rho_xy / rng**2
])
dH_el   = np.hstack([(C_sez.T @ del_sez.T).T, np.zeros((N, 3))])  # (N, 6)

H = np.vstack([dH_rng, dH_az, dH_el])   # (3N, 6) — same layout as current loop output
```

Apply the same approach to range-rate Jacobian (additional velocity-dependent rows).

**Constraint:** Output `(H, residuals)` tuple must be numerically identical to the loop version. Confirm with `test_measurements.py` — all existing tests must pass without tolerance changes.

**Estimated gain:** 2–4× Jacobian assembly for arcs with > 100 epochs; removes N Python-level function calls.

---

## TASK-P5 — SR-UKF Sigma-Point STM Linearization

**File:** `python_port/lunar_od/filters.py`  
**Function:** `square_root_ukf_predict(...)` (or equivalent predict step)  
**Current cost:** 2n+1 = 13 full ODE integrations per predict step.  
**Target:** 2 ODE calls (1 state + 1 augmented); sigma points linearized via STM.

### Change

Add `use_stm_linearization: bool = True` keyword argument.

```python
if use_stm_linearization:
    # --- 2 ODE calls ---
    # 1. Propagate mean sigma point (6-state)
    x_mean_new = propagate_state([t0, t1], x_mean, ...)[-1]

    # 2. Propagate augmented state to get STM
    x_aug0 = np.concatenate([x_mean, np.eye(6).reshape(-1, order='F')])
    x_aug  = propagate_augmented_state([t0, t1], x_aug0, ...)
    phi    = x_aug[-1, 6:].reshape(6, 6, order='F')    # (6, 6)

    # 3. Linearize all sigma points: (2n+1, 6)
    offsets       = sigma_pts - x_mean[None, :]         # (2n+1, 6)
    sigma_pts_new = x_mean_new[None, :] + (phi @ offsets.T).T

else:
    # existing path: propagate each sigma point independently
    sigma_pts_new = np.stack([
        propagate_state([t0, t1], sp, ...)[-1]
        for sp in sigma_pts
    ])

# Step 4 onward (covariance update) — unchanged
```

**Accuracy bound:** Valid when sigma-point spread α is small (α ≤ 1e-2, default 1e-3). Write a single-step test comparing `use_stm_linearization=True` vs `False` — state difference must be < 1e-3 m for a 240 s predict interval at typical lunar orbit.

**Estimated gain:** 11 fewer ODE integrations per predict step → ~10× reduction in SR-UKF wallclock (predict step dominates).

---

## TASK-P6 — Cold-Start Arc Parallelization

**File:** `python_port/lunar_od/scenarios.py`  
**Functions:** `run_batch_arc_sequence(...)`, `run_srif_arc_sequence(...)`  
**Current cost:** Serial arc loop even when arcs are fully independent.  
**Target:** `ProcessPoolExecutor` over cold-start arcs.

### Change

Add `parallel: bool = False` keyword argument to both functions.

```python
# At top of file (module level — required for pickle):
def _run_arc_worker(args: tuple) -> ArcResult:
    arc, estimator_kwargs = args
    return _estimate_single_arc(arc, **estimator_kwargs)
```

In the arc loop:
```python
if parallel:
    if start_mode != "cold":
        raise ValueError("parallel=True is only valid for start_mode='cold'.")
    from concurrent.futures import ProcessPoolExecutor
    import os
    tasks = [(arc, estimator_kwargs_for(arc)) for arc in prepared_arcs]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        arc_results = list(pool.map(_run_arc_worker, tasks))
else:
    arc_results = [_estimate_single_arc(arc, **estimator_kwargs_for(arc))
                   for arc in prepared_arcs]
```

`estimator_kwargs_for(arc)` must return a fully self-contained plain-data dict (no lambdas, no SPICE state — SPICE must be re-loaded inside the worker if needed).

**Constraint:** `_run_arc_worker` must be a module-level function (not a closure). All arguments must be picklable. If SPICE kernels are used, call `spice_loader.load_spice_kernels()` at the top of `_run_arc_worker`.

**Test:** `test_cold_start_parallel_matches_serial` — run 4-arc cold-start campaign with `parallel=False` and `parallel=True`, assert `arc_result.estimated_state` arrays are equal to 1e-10.

**Estimated gain:** ~(N_cores − 1)/N_cores × campaign wallclock for cold-start. On 4 cores with 20 arcs: ~3.5× speedup.

---

## Execution Order & Test Gates

| # | Task | Primary file | Estimated gain | Test gate |
|---|------|-------------|----------------|-----------|
| P1 | Analytical J₂ gradient | `dynamics.py` | 5–10% STM cost (J₂ on) | `test_dynamics.py` |
| P2 | Numba J₂ fast path | `dynamics.py` / `accelerated.py` | 5–10× ODE (J₂ on) | `test_dynamics.py` + spot benchmark |
| P3 | STM cache for BLS/SRIF | `estimators.py` | ~9× fewer augmented propagations | `test_estimators.py` |
| P4 | Vectorized Jacobian | `measurements.py` | 2–4× Jacobian assembly | `test_measurements.py` |
| P5 | SR-UKF STM linearization | `filters.py` | ~10× UKF predict step | `test_filters.py` |
| P6 | Cold-start parallelization | `scenarios.py` | ~3–4× campaign wallclock | new `test_scenarios.py` test |

**Rule:** After every task, run:
```bash
python -m pytest python_port/tests/ -q
```
Output must show `172 passed` (or higher if new tests were added). Do not start the next task with any failure.
