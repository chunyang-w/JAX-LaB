# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JAX-LaB is a differentiable, accelerated 2D/3D Lattice Boltzmann Method (LBM) library for multiphase and multiphysics flows, written in JAX. It extends [XLB](https://github.com/Autodesk/XLB) with Shan-Chen multiphase support, multiple equations of state, and a thermal solver. The library targets multi-GPU/TPU execution via JAX's shard_map and NamedSharding.

## Commands

### Setup
```bash
# Install JAX (choose appropriate backend)
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install dependencies (managed with uv, lockfile present)
pip install pyvista numpy matplotlib Rtree trimesh jmp orbax-checkpoint termcolor h5py
# or: pip install -r requirements.txt

export PYTHONPATH=.
```

### Running examples
```bash
python3 examples/singlephase/cavity2d.py
python3 examples/singlephase/cylinder2d.py
python3 examples/performance/MLUPS2d.py
```

### Linting
```bash
ruff check src/
ruff check examples/
```

## Architecture

### Class hierarchy

```
LBMBase (src/base.py)
├── BGKSim (src/models.py)       — BGK collision operator
├── KBCSim (src/models.py)       — KBC (entropic) collision operator
├── MRTSim (src/models.py)       — MRT collision operator
├── CascadedSim (src/models.py)  — Cascaded (central-moment) collision
├── Multiphase (src/multiphase.py) — Shan-Chen multiphase, SCMP & MCMP
│   └── (user simulation classes inherit from these)
└── Thermal (src/thermal.py)     — Thermal LBM (single-phase + multiphase)
```

Users create simulations by subclassing one of the concrete models and overriding:
- `set_boundary_conditions()` — append BC objects to `self.BCs`
- `initialize_macroscopic_fields()` — return `(rho0, u0)` arrays
- `output_data(**kwargs)` — handle I/O at each `io_rate` step
- `get_force()` — return body force delta_u (optional)

### Core modules

| File | Purpose |
|---|---|
| `src/base.py` | `LBMBase`: sharding setup, streaming, equilibrium, time loop (`run()`), checkpointing |
| `src/lattice.py` | `Lattice`, `LatticeD2Q9`, `LatticeD3Q19`, `LatticeD3Q27` — lattice vectors, weights, moments |
| `src/models.py` | Collision operators: BGK, KBC, MRT, Cascaded |
| `src/multiphase.py` | `Multiphase`: Shan-Chen pseudopotential, multi-component pytree support, wetting |
| `src/thermal.py` | `Thermal`: coupled fluid+temperature solver |
| `src/boundary_conditions.py` | All BC classes: `BounceBack`, `BounceBackHalfway`, `EquilibriumBC`, `ZouHeBC`, `RegularizedBC`, `ExtrapolationOutflow`, `ConvectiveOutflow`, `InterpolatedBounceBackBouzidi`, etc. |
| `src/eos.py` | Equation of state classes: `CarnahanStarling`, `PengRobinson`, `RedlichKwong`, `RedlichKwongSoave`, `VanDerWaals` |
| `src/utils.py` | I/O helpers: `save_image()`, `save_fields_vtk()`, `save_fields_hdf5()`, `save_BCs_vtk()`, `downsample_field()` |

### Key design patterns

**Simulation kwargs:** All simulation parameters are passed as keyword arguments to the constructor. Required keys: `lattice`, `omega`, `nx`, `ny`, `nz` (set to `0` for 2D). Optional: `precision` (default `"f32/f32"`), `io_rate`, `print_info_rate`, `checkpoint_rate`, `checkpoint_dir`, `restore_checkpoint`, `downsampling_factor`, `return_fpost`, `compute_MLUPS`.

**Precision policy:** Specified as `"compute/store"` strings (`"f32/f32"`, `"f32/f16"`, `"f64/f64"`, etc.) and enforced via `jmp.Policy`.

**Domain decomposition:** The x-axis is sharded across GPUs using `shard_map`. `nx` is automatically rounded up to a multiple of `nDevices`. Halo exchange uses `lax.ppermute` for left/right neighbor communication.

**Multiphase pytrees:** The `Multiphase` class stores per-component quantities (distribution functions, densities, velocities) as JAX pytrees (lists). Each component gets its own BCs and EOS. The number of components is set via `n_components`.

**Boundary conditions:** BCs are instantiated with `(indices, gridInfo, precisionPolicy)`. `gridInfo` is a dict with keys `nx, ny, nz, dim, lattice`. Indices are tuples of coordinate arrays. BCs with `isSolid=True` contribute to the grid mask for streaming.

**Step sequence:** `collision → apply_bc("PostCollision") → streaming → apply_bc("PostStreaming")`

### Linting rules

Line length: 150. Ruff config is in `ruff.toml`. F401 (unused imports) and F841 (unused variables) are ignored. `docs/` is excluded.

### Multi-GPU distributed runs

Call `jax.distributed.initialize()` before any JAX computation when running across multiple hosts. Single-node multi-GPU is handled automatically via `jax.device_count()`.
