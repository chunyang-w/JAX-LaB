# displacement_studies.py

Drainage/displacement simulation for a Berea sandstone sample using a two-component Shan-Chen multiphase lattice Boltzmann method with Multiple Relaxation Time (MRT) collision.

The script runs three sequential studies, each building on the previous:

1. **Droplet calibration** — tunes fluid-fluid interaction parameters and verifies surface tension via the Laplace pressure test.
2. **Droplet-on-wall calibration** — tunes wetting parameters (contact angle) against a spherical wall.
3. **Porous media drainage** — simulates CO2 displacing water through a 256^3 Berea sandstone geometry.

---

## Physics

### Fluid components

| Component | Role           |
|-----------|----------------|
| Water     | Wetting phase  |
| CO2       | Non-wetting phase (injected) |

Both fluids are modelled with the **Peng-Robinson equation of state**. Phase separation and immiscibility are controlled by the Shan-Chen inter-component coupling matrix `g_kkprime`.

### Collision operator

MRT (Multiple Relaxation Time) on a **D3Q19** lattice. The transformation matrix `M` and relaxation rates (`s_rho`, `s_e`, `s_eta`, `s_j`, `s_q`, `s_v`, `s_m`, `s_pi`) follow Coveney et al. (2002).

### Boundary conditions

**Full-way bounce-back** (`BounceBack`) is used at solid walls, extended with a wetting model parameterised by:
- `theta` — contact angle at each wall node
- `phi` — wetting strength parameter
- `delta_rho` — density correction at the wall

---

## Input

### Physics parameters (hard-coded in `__main__`)

| Parameter     | Description                                              |
|---------------|----------------------------------------------------------|
| `rho_w_l/g`   | Water liquid / gas (dissolved) equilibrium densities     |
| `rho_c_l/g`   | CO2 liquid / gas equilibrium densities                   |
| `tau_w`, `tau_c` | Relaxation times for water and CO2                    |
| `g_kkprime`   | Inter-component coupling matrix (controls surface tension)|
| `a`, `b`, `pr_omega` | PR-EOS coefficients for each component          |
| `T`           | Reduced temperature (lattice units)                      |
| `C_l`, `C_s`, `C_rho` | SI-to-lattice unit conversion factors           |

### Geometry file (Stage 3 only)

```
./assets/374_09_03_256.mat
```

An HDF5/MATLAB file containing a 256x256x256 binary array under the key `bin`:
- `1` = solid grain voxel
- `0` = pore voxel

Source: Digital Rocks Portal DOI [10.17612/93pd-y471](https://www.doi.org/10.17612/93pd-y471).

---

## Simulation stages and output

### Stage 1: Droplet3D — Laplace pressure test

Runs six independent simulations, one for each droplet radius `r in [25, 30, 35, 40, 45, 50]` (voxels), on a 150x150x150 domain.

Each run outputs VTK files to `output_{r}/data/`:

| Field       | Description                        |
|-------------|------------------------------------|
| `p`         | Total pressure                     |
| `p_water`   | Water component pressure           |
| `p_CO2`     | CO2 component pressure             |
| `rho`       | Total density                      |
| `rho_water` | Water density                      |
| `rho_CO2`   | CO2 density                        |
| `ux/uy_water` | Water velocity components        |
| `ux/uy_CO2`   | CO2 velocity components          |
| `ux/uy`       | Total velocity components        |

Console output per `io_rate` step (every 10 000 steps):
```
Spurious currents: <max velocity magnitude>
Pressure difference for radius = <r>: <delta_p>
%Error CO2 Min: <err_g>  Max: <err_l>
%Error Water Min: <err_g>  Max: <err_l>
```

The Laplace pressure difference should scale as `2*sigma/r` to verify surface tension `sigma`.

### Stage 2: DropletOnWall3D — contact angle calibration

Runs once on a 150x150x150 domain with a hemispherical wall geometry (sphere of radius 38 centred near the bottom of the box). Contact angle `pi/6` (~30 deg) is imposed on wall nodes via `phi_w = 1.17`.

Output: VTK files to `output_{r}/data/` with the same fields as Stage 1 (minus total pressure `p`).

Console output is identical to Stage 1.

### Stage 3: PorousMedia — drainage simulation

Domain: `(256 + buffer + 4) x 256 x 256` = **312 x 256 x 256** voxels, where `buffer = 52` voxels of pure CO2 act as an inlet reservoir.

The simulation applies a body force `[1e-4, 0, 0]` (x-direction) to drive CO2 through the pore space. Steady-state co-existence densities from Stage 1 are used.

Output: VTK files to `output/data/` every 1 000 steps:

| Field        | Description                              |
|--------------|------------------------------------------|
| `rho`        | Total density (interior only, 1:-1 crop) |
| `ux/uy/uz`   | Total velocity                           |
| `ux/uy/uz_CO2` | CO2 velocity                           |
| `ux/uy/uz_water` | Water velocity                       |
| `rho_water`  | Water density                            |
| `rho_CO2`    | CO2 density                              |
| `p_water`    | Water pressure                           |
| `p_CO2`      | CO2 pressure                             |
| `phi_w`      | Wetting parameter field                  |
| `theta_c`    | CO2 contact angle field                  |
| `flag`       | Solid mask (streamed)                    |

All spatial fields are cropped by 1 voxel on each face (`1:-1`) to exclude bounce-back halo nodes.

---

## How to run

### Prerequisites

```bash
# From the repository root
export PYTHONPATH=.

# Place the geometry file at:
#   examples/multiphase/MRT/assets/374_09_03_256.mat
```

### Single-GPU / single-node

```bash
cd examples/multiphase/MRT
python3 displacement_studies.py
```

### Multi-GPU (single node)

JAX detects all available GPUs automatically. No code changes are needed for single-node multi-GPU runs.

```bash
cd examples/multiphase/MRT
XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 displacement_studies.py
# or just run normally if CUDA devices are visible
python3 displacement_studies.py
```

### Expected output structure

```
examples/multiphase/MRT/
├── output_25/data/*.vti      # Stage 1, r=25
├── output_30/data/*.vti
├── ...
├── output_50/data/*.vti
├── output_38/data/*.vti      # Stage 2 (droplet on wall)
└── output/data/*.vti         # Stage 3 (porous media drainage)
```

---

## References

1. Coveney, P. V. et al. *Multiple-relaxation-time lattice Boltzmann models in three dimensions.* Phil. Trans. R. Soc. Lond. A **360**, 437-451 (2002).
2. Santos, E. J. et al. *3D Dataset of Simulations.* Digital Rocks Portal. https://doi.org/10.17612/93pd-y471
