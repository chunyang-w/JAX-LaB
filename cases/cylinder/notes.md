# DFG Benchmark 2 — Flow Around a Cylinder (Re=100)

Reference: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html

---

## Overall Strategy

The DFG Benchmark 2 is the canonical 2D unsteady cylinder flow problem. The physical setup:

- **Domain:** 2.2 m × 0.41 m channel
- **Cylinder:** center at (0.2, 0.2), diameter D = 0.1 m
- **Inflow:** parabolic (Poiseuille) profile, U_mean = 1 m/s → Re = 100 (unsteady, von Kármán shedding)
- **Target outputs:** C_D, C_L (max over time), Strouhal number St, pressure difference Δp

JAX-LaB's `examples/singlephase/cylinder2d.py` is ~90% there for this benchmark. The gaps are: pressure tap extraction, Strouhal via FFT of C_L(t), and verifying exact cylinder placement.

---

## Boundary Conditions

| Physical BC      | LBM BC in JAX-LaB               | Location       |
|------------------|----------------------------------|----------------|
| Parabolic inflow | `Regularized(..., "velocity", vel_inlet)` | left wall |
| Outflow          | `ExtrapolationOutflow`           | right wall     |
| Channel walls    | `Regularized(..., "velocity", 0)`| top + bottom   |
| Cylinder surface | `InterpolatedBounceBackBouzidi`  | cylinder cells |

`InterpolatedBounceBackBouzidi` uses the exact signed distance to the cylinder surface to achieve 2nd-order accuracy on the curved boundary — important for accurate C_D/C_L.

---

## Unit Conversion — The Core Idea

LBM does not work in SI units. It works in dimensionless **lattice units** where:
- lattice spacing Δx = 1
- timestep Δt = 1
- fluid density ρ ≈ 1

You must map the physical problem into lattice space, preserving the dimensionless numbers that govern the flow (primarily Re).

---

### What is D_physical = 0.1?

`D_physical` is the cylinder diameter as specified by the benchmark: **0.1 metres**. It is not derived — it is the benchmark's definition of the geometry.

You then choose how many lattice cells to use to represent that diameter:

```
N = D_lattice   (e.g., 40, 80)
```

This is your **resolution parameter**. Everything else is derived from it:

```
Δx_physical = D_physical / N = 0.1 / N   [m/cell]
```

The full domain in lattice units:

```
nx = round(2.2 / 0.1 * N) = 22 * N
ny = round(0.41 / 0.1 * N) ≈ 4.1 * N
```

---

### What is prescribed_vel?

`prescribed_vel` is the **mean inflow velocity in lattice units** (i.e., lattice cells per timestep). It is the LBM analog of U_mean = 1 m/s.

You choose it — it is a free parameter, subject to one critical constraint (see below). A common choice:

```python
prescribed_vel = 0.003 * (80 / N)
```

The `80/N` factor keeps the lattice velocity constant as you refine the mesh, so that Re and Ma both stay fixed across resolutions. At N=80, `prescribed_vel = 0.003`.

The Poiseuille profile peaks at U_max = 1.5 * U_mean, so the peak lattice velocity is `1.5 * prescribed_vel`.

---

### Why must u_LBM << 1? (Mach number constraint)

In LBM, the speed of sound is fixed at:

```
cs = 1/√3 ≈ 0.577  (in lattice units)
```

The Mach number is:

```
Ma = u_LBM / cs
```

The LBM Chapman-Enskog expansion (which makes LBM approximate the Navier-Stokes equations) is a low-Mach expansion. It introduces errors of order Ma². If Ma is not small, LBM does not converge to incompressible Navier-Stokes — it converges to something with artificial compressibility effects.

**Intuition:** LBM particles travel at discrete speeds. At low Ma, the flow perturbation is a tiny ripple on top of the particle distribution. At high Ma, the perturbation is large and the Taylor expansion breaks down. You can think of it like linearizing sin(x) ≈ x — works fine for small x, fails for large x.

**Rule of thumb:** keep u_LBM < 0.1, ideally < 0.05. This is why `prescribed_vel = 0.003` — comfortably small.

---

### How to determine ω (omega) and ν_LBM

Step 1: **Fix Re** (given by the benchmark, Re = 100).

Step 2: Use the definition of Re in lattice units:

```
Re = U_mean_LBM * D_LBM / ν_LBM
→  ν_LBM = prescribed_vel * N / Re
```

Step 3: In BGK-LBM (single relaxation time), the kinematic viscosity relates to the relaxation time τ:

```
ν_LBM = cs² * (τ - 0.5) = (τ - 0.5) / 3
```

Solving for τ:

```
τ = 3 * ν_LBM + 0.5
```

And omega is simply:

```
ω = 1 / τ = 1 / (3 * ν_LBM + 0.5)
```

In code:

```python
visc = prescribed_vel * diam / Re
omega = 1.0 / (3.0 * visc + 0.5)
```

**Stability constraint:** τ > 0.5 (i.e., ω < 2) is required for stability. τ too close to 0.5 (ω → 2) makes the simulation unstable. τ = 1 (ω = 1) is a common stable choice.

---

### The prescribed_vel tradeoff: accuracy vs. stability vs. efficiency

Many values of `prescribed_vel` represent the same Re=100 physics — Re is dimensionless, and as long as `vel * N / Re` gives the right ν_LBM, the physical simulation is identical. But the choice has three simultaneous consequences:

**1. Mach number errors (accuracy)**
```
Ma = vel / cs = vel * √3
error ~ O(Ma²) ~ O(vel²)
```
Smaller vel → smaller artificial compressibility errors → more accurate.

**2. τ approaching 0.5 (stability)**
```
τ = 3 * (vel * N / Re) + 0.5
```
As vel → 0, τ → 0.5, ω → 2. This is the BGK stability boundary. Too small a vel makes the simulation unstable.

**3. Number of timesteps (efficiency)**
Each timestep represents a physical time ∝ vel (at fixed N). Halving vel doubles the number of steps needed.

The three constraints create a sweet spot:
```
vel too small ──── τ → 0.5 ──── unstable
                      ↑
               sweet spot
                      ↓
vel too large ──── Ma errors ──── wrong physics
```

`prescribed_vel = 0.003 * (80/N)` lands in this sweet spot: Ma ≈ 0.005, τ ≈ 0.85.

MRT and KBC models decouple stability from τ (non-hydrodynamic modes get their own relaxation rates), so they can run at lower vel (higher accuracy) without instability.

---

### Does ω change if you switch collision models?

**Yes and no** — the value of ω carries the same physical meaning (it encodes ν_LBM, which encodes Re), but *how it is used* differs by model:

| Model | How ω is used |
|-------|--------------|
| **BGK** (single relaxation time) | One ω relaxes all modes. Simple, but unstable near τ = 0.5 (high Re). |
| **MRT** (multiple relaxation time) | Each moment (density, momentum, stress, energy, ...) gets its own relaxation rate. The physical viscosity ω is still used for the stress modes; the others are free parameters tuned for stability. |
| **KBC** (entropic) | Uses a single ω but adapts it locally to maximize entropy — acts as a stabilizer. The base ω still comes from Re via the same formula. |
| **Cascaded** | Similar to MRT but in central-moment space. Again, the stress-mode relaxation rate is ω from Re; others are free. |

**Practical implication:** you compute ω from Re using the same formula regardless of model. For MRT/Cascaded, you then *also* specify the free relaxation rates for non-hydrodynamic modes (they affect stability, not the physical viscosity).

---

## Force and Coefficient Calculation

```python
boundary_force = cylinder.momentum_exchange_force(f_poststreaming, f_postcollision)
drag = boundary_force[0]
lift = boundary_force[1]

# Non-dimensionalize (ρ = 1 in LBM)
C_D = 2 * drag / (U_mean_lb**2 * D_lattice)
C_L = 2 * lift / (U_mean_lb**2 * D_lattice)
```

The **momentum exchange method** counts the momentum transferred to the solid at each timestep via bounce-back.

**Strouhal number** from FFT of C_L(t):

```python
St = f_shedding * D / U_mean
# f_shedding from peak of FFT of C_L time series (in physical units)
```

**Pressure difference** at taps (0.15, 0.2) and (0.25, 0.2):

```python
# In LBM: p = cs² * ρ = ρ/3
delta_p = (rho[x_A, y_A] - rho[x_B, y_B]) / 3.0
```

---

## LBMBase Framework Hooks (base.py)

Three methods in `LBMBase` are stubs designed for you to override in your subclass.

### `set_boundary_conditions`

Called once, during `__init__`, inside `_create_boundary_data()`. Your job is to append BC objects to `self.BCs`. After you return, the framework automatically:
1. Collects all BCs with `isSolid=True` to build the streaming grid mask
2. Calls `bc.create_local_mask_and_normal_arrays(grid_mask)` on every BC

You declare *what* and *where*. The framework handles the geometry preprocessing.

### `output_data(**kwargs)`

Called by `handle_io_timestep` inside the main `run()` loop, whenever `timestep % ioRate == 0`. All arrays arrive as **numpy arrays on the host** (already gathered from GPU), so you can use plain numpy/matplotlib/VTK freely.

The kwargs dict:

| Key | Shape | Contents |
|-----|-------|----------|
| `timestep` | int | current step number |
| `rho` | (nx, ny, 1) | density at current step |
| `rho_prev` | (nx, ny, 1) | density at previous io step |
| `u` | (nx, ny, 2) | velocity at current step |
| `u_prev` | (nx, ny, 2) | velocity at previous io step |
| `f_poststreaming` | (nx, ny, q) | populations after streaming |
| `f_postcollision` | (nx, ny, q) | populations after collision — **only non-None if `return_fpost=True`** |

`rho_prev`/`u_prev` exist so you can compute convergence error without managing state yourself. `f_postcollision` is needed for the momentum exchange force (drag/lift) — that's why the cylinder example sets `return_fpost=True`.

### `initialize_macroscopic_fields`

Called once inside `assign_fields_sharded`, which is the first thing `run()` does. It returns `(rho0, u0)`. If you return `(None, None)` (the default), the distribution functions are initialized to the lattice weights `w` — equivalent to **ρ=1, u=0 everywhere** (fluid at rest).

**Why the cylinder example does not override it:** starting from rest is physically valid. The Poiseuille inflow BC drives the flow from the left boundary every timestep, and the flow develops naturally. The code already discards the first half of the run before collecting statistics, which absorbs the transient.

You *would* override it when:
- Pre-loading a nearly-converged state to shorten the transient
- Multiphase problems that need a seeded density field (e.g., two fluids already separated)
- Restarting from a saved numpy array (separate from the checkpoint system)

---

## Simulation Procedure and Data Structures

### The central array: `f`

```
f : shape (nx, ny, 9)
     │    │    └─ 9 velocity directions (D2Q9)
     │    └─ ny lattice nodes in y
     └─ nx lattice nodes in x
```

`f[i, j, k]` = probability density at node (i,j) for direction k. D2Q9 directions:

```
direction:  0   1   2   3   4   5   6   7   8
cx       :  0   1   0  -1   0   1  -1  -1   1
cy       :  0   0   1   0  -1   1   1  -1  -1

        6  2  5
         \ | /
      3 - 0 - 1
         / | \
        7  4  8
```

`rho` and `u` are always **derived** from `f`, never stored independently:

```
rho[i,j]   = sum(f[i,j,:])                  shape (nx, ny, 1)
u[i,j]     = sum(f[i,j,:] * c) / rho        shape (nx, ny, 2)
```

`f` is the only state. Everything else is recomputed on demand.

### Setup (once, in `__init__`)

- `LatticeD2Q9` builds `c` (2,9), `w` (9,), `cc` (9,3)
- `set_boundary_conditions()` populates `self.BCs` list
- `create_grid_mask()` builds a `(nx+halo, ny+halo, 9)` bool mask marking nodes that stream into solids

### Initialization (`run()`)

```python
f = assign_fields_sharded()
# → f[i,j,:] = w for all nodes  (ρ=1, u=0 everywhere)
```

### One timestep: `step(f, timestep)`

```
f (nx,ny,9)
    │
    ▼ A. collision  (BGK: models.py:24)
    │     rho, u = update_macroscopic(f)
    │     feq = rho * w * (1 + 3(u·c) + 4.5(u·c)² - 1.5|u|²)
    │     fout = f - ω * (f - feq)       ← one line of physics
    │
    ▼ B. apply_bc PostCollision  [no-op for cylinder]
    │
    ▼ C. streaming  (base.py:711)
    │     for each direction k:
    │       f[:,:,k] = jnp.roll(f[:,:,k], (cx[k], cy[k]), axis=(0,1))
    │     + halo exchange via lax.ppermute for multi-GPU
    │
    ▼ D. apply_bc PostStreaming
    │     BCs[0] InterpolatedBounceBackBouzidi → no-slip on cylinder
    │     BCs[1] ExtrapolationOutflow          → absorbing outlet
    │     BCs[2] Regularized (inlet)           → inject Poiseuille profile
    │     BCs[3] Regularized (walls)           → zero-velocity walls
    │
    ▼ f (next timestep, same shape (nx,ny,9))
```

### BGK collision detail

The entire fluid physics reduces to three lines:
```python
feq = rho * w * (1.0 + cu*(1.0 + 0.5*cu) - usqr)   # equilibrium
fneq = f - feq                                        # deviation
fout = f - omega * fneq                               # relax toward feq
```

`omega=1` → instant equilibration (maximum viscous dissipation).
`omega→2` → instability boundary (τ→0.5, zero viscosity).

### I/O: `output_data`

Every `ioRate` steps:
1. `update_macroscopic(f)` → `rho`, `u` on device
2. `process_allgather` → numpy arrays on host
3. `output_data(**kwargs)` called — in the cylinder case computes drag/lift via momentum exchange on `BCs[0]`

---

## What Needs to Be Added to cylinder.py

1. Verify cylinder placement: benchmark uses cy = 0.2/0.1 * D = 2D from bottom — already correct
2. Extract `rho` at the two pressure taps in `output_data`
3. Collect `C_L(t)` array over time, then FFT after the run for Strouhal
4. Run long enough: ~10+ shedding cycles for converged statistics (the code already waits until `timestep > 0.5 * niter_max`)
