# Drainage / Imbibition — SWCC

Soil-water characteristic curve (SWCC) simulation using a two-component Shan-Chen LBM (BGK).
Accepts any porous geometry and infers the domain size directly from the input file.

## Usage

```bash
# From the repo root
export PYTHONPATH=.

python3 cases/drainage/characteristic_curves.py --geometry /path/to/geometry.npy
python3 cases/drainage/characteristic_curves.py --geometry /path/to/geometry.mat --simulation imbibition
python3 cases/drainage/characteristic_curves.py --geometry geom.npy --buffer 8 --timesteps 100000 --io-rate 500
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--geometry` | *(required)* | Path to geometry file (`.npy` or `.mat`, `True`/`1` = solid) |
| `--simulation` | `drainage` | `drainage` or `imbibition` |
| `--drho` | `0.0092` | Density perturbation at inlet/outlet (pressure driving force) |
| `--width` | `4` | Liquid-vapour interface width in lattice units |
| `--buffer` | `8` | Buffer planes prepended/appended along x for inlet/outlet |
| `--timesteps` | `250000` | Total simulation timesteps |
| `--io-rate` | `1000` | Output interval (VTK + Pc-S line written every N steps) |

## Geometry format

| Format | Convention |
|---|---|
| `.npy` | `bool` or `int` array, shape `(Gx, Gy, Gz)`, `True`/`1` = solid grain |
| `.mat` | HDF5 file with dataset key `"bin"`, value `1` = solid grain |

The effective simulation domain is `(Gx + 2·buffer) × Gy × Gz`.

## Output

- `output_<simulation>/` — VTK files with fields `rho`, `rho_water`, `rho_air`, `p_water`, `p_air`, `ux/uy/uz`, `flag`
- `characteristic_curve_<simulation>.txt` — CSV with columns `Capillary Pressure, Saturation`
- `images/` — rendered PNG snapshots at each `io_rate`

## Physics parameters

All fluid parameters are calibrated and scale-independent:

| Parameter | Value | Description |
|---|---|---|
| `g_kkprime[0,1]` | `0.54` | Water–air interaction strength |
| `rho_w_l / rho_w_g` | `2.0 / 0.1` | Water liquid/gas densities |
| `rho_a_l / rho_a_g` | `2.0 / 0.1` | Air liquid/gas densities |
| `omega` | `1.0` | Relaxation frequency (τ = 1) for both components |
| `theta_w` (grain) | `π/6` | Water contact angle on solid (30°, water-wet) |
| `delta_rho_a` (grain) | `0.2` | Wetting density perturbation for air on solid |
