# Characteristic Curves: Soil-Water Retention Simulation

This script computes the **soil-water characteristic curve (SWCC)** — also called the capillary pressure–saturation (Pc–S) relationship — for a realistic 256³ sphere pack using a two-component (water + air) **Shan-Chen multicomponent multiphase (MCMP)** Lattice Boltzmann model with BGK collision.

The SWCC describes how strongly a porous medium retains water as a function of saturation. It is fundamental to unsaturated flow modelling (Richards equation), geomechanics, and reservoir engineering.

The workflow is a **three-stage calibration-then-production pipeline**:

```
Stage 1: Droplet3D          → calibrate surface tension  (g_kkprime, rho_l, rho_g)
Stage 2: DropletOnWall3D    → calibrate contact angle    (theta_w, delta_rho_w/a)
Stage 3: PorousMedia        → generate Pc-S curves       (imbibition + drainage)
```

---

## Prerequisites

```bash
# Install phantomgaze (volumetric renderer used for images)
pip install phantomgaze

# The porous geometry file must be present:
#   examples/multiphase/BGK/assets/374_05_03_256.mat
# Download from Digital Rocks Portal (DRP-372):
# https://digitalporousmedia.org/published-datasets/tapis/projects/drp.project.published/drp.project.published.DRP-372/374_05_03/374_05_03_256/
```

---

## Running

```bash
cd /path/to/JAX-LaB
export PYTHONPATH=.
python3 examples/multiphase/BGK/characteristic_curves.py
```

By default (`RUN_CALIBRATION = False` at the top of `__main__`) **only Stage 3 runs** — the two calibration stages are skipped because their parameters are already baked into the script.  Set `RUN_CALIBRATION = True` only when you need to re-validate the surface tension or contact angle from scratch (e.g. after changing `g_kkprime` or `tau`).

The script produces:

| Output | Description |
|---|---|
| `surface_tension.txt` | Radius vs. ΔP data for Young-Laplace verification |
| `output_{r}/` | VTK fields for each droplet radius |
| `output/` | VTK fields for the contact angle droplet-on-wall test |
| `characteristic_curve_imbibition.txt` | Pc, S time series during imbibition |
| `characteristic_curve_drainage.txt` | Pc, S time series during drainage |
| `output_imbibition/` | VTK fields at each io step (imbibition) |
| `output_drainage/` | VTK fields at each io step (drainage) |
| `images/porous_imbibition_XXXXXXX.png` | Volume renders during imbibition |
| `images/porous_drainage_XXXXXXX.png` | Volume renders during drainage |

---

## Why do water and air share the same density values?

Looking at the parameters you might immediately notice something surprising:

```python
rho_w_l = 2.0   # water liquid density
rho_w_g = 0.1   # water gas density

rho_a_l = 2.0   # air liquid density   ← same as water!
rho_a_g = 0.1   # air gas density      ← same as water!
```

In reality, liquid water is roughly **830× denser than air** at atmospheric conditions.  Yet here both components share the same equilibrium densities.  This is not a mistake — it is a deliberate property of the **symmetric Shan-Chen multicomponent model**, and understanding it is key to interpreting the results correctly.

### Phase separation is driven by repulsion, not by density contrast

In the Shan-Chen MCMP model, the two components separate into immiscible-like phases because of the **inter-component repulsion force** controlled by `g_kkprime[0,1] = 0.54`.  When two components repel each other strongly enough, they spontaneously segregate: component 0 (water) concentrates in one region and component 1 (air) concentrates in another, even though they start with the same bulk density parameters.

The density values `rho_l` and `rho_g` in the code do not encode the physical mass density of water vs. air.  Instead, they define the **equilibrium number density of each component in its own condensed (rich) vs. dilute (lean) phase**:

| Region | Component 0 (water) | Component 1 (air) |
|---|---|---|
| Water-filled pore | ρ₀ ≈ rho_w_l = 2.0 (rich) | ρ₁ ≈ rho_a_g = 0.1 (lean) |
| Air-filled pore | ρ₀ ≈ rho_w_g = 0.1 (lean) | ρ₁ ≈ rho_a_l = 2.0 (rich) |

Each component is simultaneously present everywhere in the domain, but at very different concentrations depending on which fluid occupies that pore.  The word "water" is a label for component 0; the word "air" is a label for component 1.

### What actually distinguishes water from air in this model

The physical distinction is encoded entirely in the **wettability boundary conditions**, not in the bulk densities:

- **Contact angle:** `theta_w = π/6` (30°) makes component 0 water-wet — it prefers the solid surface.  `theta_a = π - π/6` (150°) makes component 1 non-wetting.
- **Preferential adsorption:** `delta_rho_a = 0.2` on solid nodes slightly enriches the air component near the wall, counteracting its non-wetting nature and giving a realistic finite contact angle.

The capillary pressure, invasion order, and residual saturation all emerge from these wetting contrasts, not from a density ratio.

### What this model cannot capture

Because the physical density ratio (water/air ≈ 830) is not reproduced, phenomena that depend on it are outside the scope of this simulation:

- **Gravity / buoyancy** — body forces proportional to density difference are not meaningful here.
- **Viscosity ratio effects** — the viscosities are equal (`tau_w = tau_a = 1.0`), so the model does not capture the M ≫ 1 mobility ratio of real water/air systems.  This affects relative permeability curves but not the Pc–S curve shape at capillary-dominated conditions.
- **Compressibility contrasts** — the compressibility of both phases is the same in this symmetric formulation.

For capillary-dominated pore-scale flow — which is exactly what the SWCC measures — these are secondary effects.  The dominant physics (surface tension, contact angle, pore geometry, invasion percolation sequence) are all correctly represented.

### Summary

> The equal densities are a **model simplification**, not a physical error.  "Water" and "air" are labels for the wetting and non-wetting components respectively.  The physics that controls the SWCC — surface tension and contact angle — is fully captured; the real mass density ratio is intentionally sacrificed for numerical simplicity and stability.

---

## Lattice units and key parameters

| Symbol | Value | Meaning |
|---|---|---|
| `rho_w_l` | 2.0 | Water liquid-phase density |
| `rho_w_g` | 0.1 | Water gas-phase (dissolved) density |
| `rho_a_l` | 2.0 | Air liquid-phase density |
| `rho_a_g` | 0.1 | Air gas-phase density |
| `tau_w = tau_a` | 1.0 | BGK relaxation time → ω = 1.0, ν = cs²(τ−0.5) = 1/6 |
| `g_kkprime[0,1]` | 0.54 | Water–air repulsion (Shan-Chen coupling) |
| `g_kkprime[0,0] = g_kkprime[1,1]` | −0.06 | Self-interaction (mild cohesion) |
| `width` | 4 | Diffuse interface half-width in lattice units |
| `drho` | 0.0092 | Inlet/outlet density perturbation driving flow |
| `buffer` | 8 | Buffer layers added each side in x to host inlet/outlet BCs |

---

## Stage 1 — `Droplet3D`: surface tension calibration

A spherical water droplet is initialised in a periodic 256³ box using a hyperbolic-tangent profile.  Four radii (25, 30, 35, 40 lu) are tested.

**What is measured:** At steady state, the pressure difference between the droplet centre and the bulk far field is compared to the Young-Laplace prediction:

```
ΔP = 2σ / R
```

Plotting ΔP vs. 1/R gives a straight line whose slope is `2σ`. This validates the fluid–fluid coupling constant `g_kkprime[0,1] = 0.54`.

**Override rationale:** `compute_potential` returns zero (no external/gravity potential) and `compute_pressure` is ideal gas `p = ρ cs²`. This is the simplest valid closure — we are not trying to model a real EOS here, only quantify the surface tension generated by the inter-component Shan-Chen force.

---

## Stage 2 — `DropletOnWall3D`: contact angle calibration

A water droplet is placed on a spherical solid wall. The solid is defined by:

```python
sphere = (x - nx/2)² + (y - ny/2)² + (z - nz/2 + R + 24)² - R²  ≤ 0
```

Wetting boundary conditions (`BounceBack` with `theta_w`, `phi_w`, `delta_rho`) are applied only on those solid nodes via `set_boundary_conditions()`. The target contact angle on the solid is **30°** (water-wet):

```python
theta_w[ind] = np.pi / 6        # 30° water contact angle
theta_a[ind] = np.pi - np.pi/6  # 150° air contact angle (complementary)
delta_rho_a  = 0.2              # Preferential adsorption strength for air
```

After equilibration you measure the contact angle from the rendered VTK output and adjust `theta_w` / `delta_rho` until it matches the target. The calibrated values are then carried forward to Stage 3.

---

## Stage 3 — `PorousMedia`: capillary pressure curves

The geometry is a 256³ sphere pack (porosity ≈ 0.381) from the Digital Rocks Portal. An 8-cell buffer is prepended and appended in x to host inlet and outlet equilibrium BCs:

```
| buffer (inlet BC) | ← 256 cells of porous medium → | buffer (outlet BC) |
```

Two processes are run back-to-back:

### Imbibition (water invades air-saturated medium)

- **Initial condition:** domain filled with gas-phase water (`rho_w_g`) and liquid-phase air (`rho_a_l`) — air occupies all pore space.
- **Inlet BC:** slightly elevated water density `rho_w_g + drho` + non-zero velocity pushes water in from the left.
- **Outlet BC:** slightly reduced density `rho_w_g − drho` to maintain the pressure gradient.
- The simulation records `(Pc, S)` every `io_rate` steps. Saturation S is the fraction of pore volume occupied by liquid water.

### Drainage (air invades water-saturated medium)

- **Initial condition:** domain filled with liquid-phase water (`rho_w_l`) and gas-phase air (`rho_a_g`) — water occupies all pore space.
- **Inlet BC:** slightly elevated water density `rho_w_l + drho` pushing water out from the right (by symmetry, air invades from the left).
- Same `(Pc, S)` recording.

Both use the same `PorousMedia` class; which branch of logic executes is controlled by the global `simulation` string (`"imbibition"` or `"drainage"`).

### Capillary pressure and saturation calculation

```python
# Inside PorousMedia.output_data():
water = (rho_water > rho_air)         # voxels where water is the denser component
air   = (rho_water < rho_air)

p_wetting    = mean(p_water[pore & water])
p_nonwetting = mean(p_air  [pore & air  ])

Pc = p_nonwetting - p_wetting         # capillary pressure (air − water)
S  = volume_water / (volume_water + volume_air)
```

---

## Visualisation

Volume renders are produced with [PhantomGaze](https://github.com/loliverhennigh/PhantomGaze) using ray-cast contours:

```python
# Imbibition: render rho_water at threshold 0.95
screen_buffer = pg.render.contour(rho_volume, threshold=0.95, colormap=red,   ...)
screen_buffer = pg.render.contour(boundary_volume, threshold=0.95, colormap=grey, ...)
```

- **Red surface** — dense phase of the tracked fluid at isosurface ρ = 0.95
- **Grey semi-transparent surface** — solid grain boundaries

---

## Output file format

`characteristic_curve_imbibition.txt` and `characteristic_curve_drainage.txt`:

```
Capillary Pressure,Saturation
0.00312,0.982
...
```

Each line is one `io_rate` snapshot. Plot Pc vs. S to obtain the SWCC. Combining both curves gives the full hysteresis loop.
