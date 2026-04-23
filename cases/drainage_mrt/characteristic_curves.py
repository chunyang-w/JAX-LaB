"""
Soil-water characteristic curve (SWCC) for a porous geometry of arbitrary size.
MRT version — identical water/air Shan-Chen physics as cases/drainage (BGK),
with MRT collision replacing BGK for improved numerical stability during
Haines-jump snap-off events.

MRT decouples the bulk-stress and energy-flux modes (s_e, s_eta, s_q) from the
shear-viscosity mode (s_v = 1/tau), so the non-hydrodynamic modes that blow up
during snap-off can be damped independently without changing fluid physics.

Collision matrix: Coveney et al. 2002 (D3Q19, same as displacement_studies.py).

Accepts geometry from either:
  .mat  (HDF5, key "bin", 0=pore 1=solid)
  .npy  (bool/int array, True/1=solid)

Domain size is inferred automatically from the geometry file.

Usage:
    python3 characteristic_curves.py --geometry /path/to/geometry.npy
    python3 characteristic_curves.py --geometry geom.mat --simulation imbibition
    python3 characteristic_curves.py --geometry geom.npy --buffer 8 --timesteps 100000 --io-rate 500
"""

import argparse
import os
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import phantomgaze as pg
import jax.numpy as jnp
from jax import jit
from jax.tree import map

from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack, EquilibriumBC
from src.utils import save_fields_vtk


class PorousMedia(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        if simulation == "imbibition":
            rho = rho_w_g * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)

            rho = rho_a_l * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
        else:
            rho = rho_w_l * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)

            rho = rho_a_g * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = map(lambda rho: rho * self.lattice.cs2, rho_tree)
        return p_tree

    # NOTE: @partial(jit) intentionally removed — see base class note on compute_total_pressure.
    def compute_total_pressure(self, p_tree, rho_tree=None):
        p_water = p_tree[0]
        p_air = p_tree[1]
        return p_water + p_air + 3 * self.g_kkprime[0, 1] * p_air * p_water

    def set_boundary_conditions(self):
        if simulation == "imbibition":
            inlet = self.boundingBoxIndices["left"]
            rho_inlet = (rho_w_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            rho_inlet = (rho_a_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))

            outlet = self.boundingBoxIndices["right"]
            rho_outlet = (rho_w_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            rho_outlet = (rho_a_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))

            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))

            grain = tuple(idx.T)
            self.BCs[0].append(BounceBack(grain, self.gridInfo, self.precisionPolicy, theta_w[grain], phi_w[grain], delta_rho_w[grain]))
            self.BCs[1].append(BounceBack(grain, self.gridInfo, self.precisionPolicy, theta_a[grain], phi_a[grain], delta_rho_a[grain]))
        else:
            inlet = self.boundingBoxIndices["left"]
            rho_inlet = (rho_w_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            rho_inlet = (rho_a_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))

            outlet = self.boundingBoxIndices["right"]
            rho_outlet = (rho_w_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            rho_outlet = (rho_a_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))

            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))

            grain = tuple(idx.T)
            self.BCs[0].append(BounceBack(grain, self.gridInfo, self.precisionPolicy, theta_w[grain], phi_w[grain], delta_rho_w[grain]))
            self.BCs[1].append(BounceBack(grain, self.gridInfo, self.precisionPolicy, theta_a[grain], phi_a[grain], delta_rho_a[grain]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, :, 1:-1, 1:-1, :])
        p_water = np.array(kwargs.get("p_tree")[0][:, 1:-1, 1:-1, :])
        p_air = np.array(kwargs.get("p_tree")[1][:, 1:-1, 1:-1, :])
        u = np.array(kwargs.get("u_total")[0, :, 1:-1, 1:-1, :])
        rho_water = np.array(kwargs.get("rho_tree")[0][0, :, 1:-1, 1:-1, :])
        rho_air = np.array(kwargs.get("rho_tree")[1][0, :, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "p_air": p_air[..., 0],
            "p_water": p_water[..., 0],
            "flag": self.solid_mask_streamed[0][:, 1:-1, 1:-1, 0],
        }
        save_fields_vtk(timestep, fields, f"output_{simulation}", "data")

        # Capillary pressure and saturation computed over the porous interior only.
        # Slice by nx_geo (raw geometry size), not self.nx (which includes both buffers).
        rho_water_int = rho_water[buffer: buffer + nx_geo, ...]
        rho_air_int = rho_air[buffer: buffer + nx_geo, ...]
        porous = np.array(self.solid_mask_streamed[0][buffer: buffer + nx_geo, 1:-1, 1:-1, 0])
        p_water_int = p_water[buffer: buffer + nx_geo, ...]
        p_air_int = p_air[buffer: buffer + nx_geo, ...]

        water = (rho_water_int > rho_air_int)[..., 0]
        air = (rho_water_int < rho_air_int)[..., 0]
        p_wetting = np.mean(p_water_int[(~porous & water) == 1])
        p_nonwetting = np.mean(p_air_int[(~porous & air) == 1])

        P_c = p_nonwetting - p_wetting
        v_w = np.sum(~porous & water)
        v_nw = np.sum(~porous & air)
        S = v_w / (v_w + v_nw)
        file.write(f"{P_c},{S}\n")
        file.flush()
        print(f"[t={timestep}] Pc={P_c:.6f}  S_water={S:.4f}  rho_w=[{rho_water_int.min():.3f},{rho_water_int.max():.3f}]  rho_a=[{rho_air_int.min():.3f},{rho_air_int.max():.3f}]")

        dx, dy, dz = (0.01, 0.01, 0.01)
        origin = (0.0, 0.0, 0.0)

        red = pg.SolidColor(color=(1.0, 0.0, 0.0), opacity=1.0)
        grey = pg.SolidColor(color=(0.439, 0.475, 0.757), opacity=0.04)

        if simulation == "imbibition":
            render_rho = rho_water_int[0: nx_geo, ..., 0]
        else:
            render_rho = rho_air_int[0: nx_geo, ..., 0]

        rho_volume = pg.objects.Volume(
            jnp.array(render_rho, dtype=self.precisionPolicy.compute_dtype),
            spacing=(dx, dy, dz),
            origin=origin,
        )
        boundary_volume = pg.objects.Volume(
            jnp.array(porous[0: nx_geo, ...], dtype=jnp.float32),
            spacing=(dx, dy, dz),
            origin=origin,
        )

        radius = max(20, nx_geo // 3)
        angle = 20 * np.pi / 180
        focal_point = (nx_geo * dx / 2, 3 * ny_geo * dy / 4, nz_geo * dz / 2)
        camera_position = (
            nx_geo * dx + radius * np.cos(angle) * dx,
            -0.1,
            -nz_geo * dz + radius * np.sin(angle) * dz,
        )
        camera = pg.Camera(
            position=camera_position,
            focal_point=focal_point,
            view_up=(0.0, -1.0, 0.0),
            height=2160,
            width=3840,
            background=pg.SolidBackground(color=(1.0, 1.0, 1.0)),
        )

        screen_buffer = pg.render.contour(rho_volume, threshold=0.95, colormap=red, camera=camera)
        screen_buffer = pg.render.contour(boundary_volume, camera, threshold=0.95, colormap=grey, screen_buffer=screen_buffer)
        screen_buffer = pg.render.wireframe(
            lower_bound=(0, 0, 0),
            upper_bound=(nx_geo * dx, ny_geo * dy, nz_geo * dz),
            color=pg.SolidColor(color=(0.0, 0.0, 0.0)),
            thickness=0.0025,
            camera=camera,
            screen_buffer=screen_buffer,
        )
        os.makedirs("images", exist_ok=True)
        plt.imsave(
            f"images/porous_{simulation}" + str(timestep).zfill(7) + ".png",
            np.minimum(screen_buffer.image.get(), 1.0),
        )


def _load_geometry(path: str) -> np.ndarray:
    """
    Load a binary porous geometry from .mat or .npy and return a boolean
    array of shape (Gx, Gy, Gz) where True = solid grain.

    .mat  — HDF5 file with dataset "bin"; value 1 = solid.
    .npy  — NumPy array of bool or int; True / 1 = solid.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(bool)
    elif ext == ".mat":
        with h5py.File(path, "r") as f:
            return np.array(f["bin"], dtype=bool)
    else:
        raise ValueError(f"Unsupported geometry format '{ext}'. Use .npy or .mat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRT SWCC — water/air Shan-Chen LBM (arbitrary geometry size)")
    parser.add_argument("--geometry", required=True, help="Path to geometry file (.npy or .mat, True/1=solid)")
    parser.add_argument("--simulation", default="drainage", choices=["drainage", "imbibition"],
                        help="Which process to simulate. Default: drainage")
    parser.add_argument("--drho", type=float, default=0.0092,
                        help="Density perturbation at inlet/outlet BCs (pressure driving force). Default: 0.0092")
    parser.add_argument("--width", type=int, default=4,
                        help="Liquid-vapour interface width in lattice units. Default: 4")
    parser.add_argument("--buffer", type=int, default=8,
                        help="Number of buffer lattice units prepended/appended along x. Default: 8")
    parser.add_argument("--timesteps", type=int, default=250000,
                        help="Total simulation timesteps. Default: 250000")
    parser.add_argument("--io-rate", type=int, default=1000,
                        help="Output/checkpoint interval in timesteps. Default: 1000")
    args = parser.parse_args()

    simulation = args.simulation
    drho = args.drho
    width = args.width
    buffer = args.buffer

    # -------------------------------------------------------------------------
    # Physics parameters — identical to cases/drainage (BGK), scale-independent
    # -------------------------------------------------------------------------
    precision = "f32/f32"
    g_kkprime = -0.06 * np.ones((2, 2))
    g_kkprime[0, 1] = 0.54
    g_kkprime[1, 0] = 0.54

    rho_w_l = 2.0
    rho_w_g = 0.1
    rho_a_l = 2.0
    rho_a_g = 0.1

    # -------------------------------------------------------------------------
    # MRT collision matrix (D3Q19, Coveney et al. 2002)
    #
    # Transforms populations to moment space where each moment relaxes
    # independently. s_v = 1/tau controls shear viscosity, identical to
    # BGK with omega=1.0. s_e, s_eta, s_q damp the energy and bulk-stress
    # modes that amplify during Haines-jump snap-off and cause BGK to diverge.
    # Values from displacement_studies.py.
    # -------------------------------------------------------------------------
    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :]  = en**0
    M[1, :]  = 19 * en**2 - 30
    M[2, :]  = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :]  = e[:, 0]
    M[4, :]  = (5 * en**2 - 9) * e[:, 0]
    M[5, :]  = e[:, 1]
    M[6, :]  = (5 * en**2 - 9) * e[:, 1]
    M[7, :]  = e[:, 2]
    M[8, :]  = (5 * en**2 - 9) * e[:, 2]
    M[9, :]  = 3 * e[:, 0]**2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0]**2 - en**2)
    M[11, :] = e[:, 1]**2 - e[:, 2]**2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1]**2 - e[:, 2]**2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1]**2 - e[:, 2]**2) * e[:, 0]
    M[17, :] = (e[:, 2]**2 - e[:, 0]**2) * e[:, 1]
    M[18, :] = (e[:, 0]**2 - e[:, 1]**2) * e[:, 2]

    # tau=1.0 for both components → s_v=1.0, same shear viscosity as BGK omega=1.0
    tau_w = 1.0
    tau_a = 1.0
    s_rho = [0.0, 0.0]   # conserved: mass
    s_j   = [0.0, 0.0]   # conserved: momentum
    s_e   = [1.0, 1.0]   # energy mode — match BGK
    s_eta = [1.0, 1.0]   # energy-stress mode — match BGK
    s_q   = [1.0, 1.0]   # heat flux — match BGK (1.1 caused over-relaxation oscillations during Haines jumps)
    s_pi  = [1.0, 1.0]   # normal stress difference
    s_m   = [1.0, 1.0]   # antisymmetric stress
    s_v   = [1 / tau_w, 1 / tau_a]   # shear viscosity — matches BGK

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------
    solid_mask = _load_geometry(args.geometry)
    nx_geo, ny_geo, nz_geo = solid_mask.shape
    print(f"Geometry loaded : {args.geometry}")
    print(f"Geometry shape  : {nx_geo} x {ny_geo} x {nz_geo}")
    print(f"Solid fraction  : {solid_mask.mean():.4f}")

    nx = nx_geo + 2 * buffer
    ny = ny_geo
    nz = nz_geo

    # Grain indices shifted past the inlet buffer in x
    raw_ind = np.where(solid_mask)
    idx = np.zeros((len(raw_ind[0]), 3), dtype=int)
    idx[:, 0] = raw_ind[0] + buffer
    idx[:, 1] = raw_ind[1]
    idx[:, 2] = raw_ind[2]

    # Wettability fields — identical to cases/drainage
    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[tuple(idx.T)] = np.pi / 6          # water-wet grains (30 deg)
    theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))

    phi_w = np.ones((nx, ny, nz, 1))
    phi_w[tuple(idx.T)] = 1.0
    phi_a = np.ones((nx, ny, nz, 1))

    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_a = np.zeros((nx, ny, nz, 1))
    delta_rho_a[tuple(idx.T)] = 0.2

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    kwargs = {
        "n_components": 2,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        "body_force": [0.0, 0.0, 0.0],
        "precision": precision,
        "M": [M, M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "s_pi": s_pi,
        "s_m": s_m,
        "kappa": [0.0, 0.0],
        "k": [1.0, 1.0],
        "A": np.zeros((2, 2)),
        "io_rate": args.io_rate,
        "print_info_rate": args.io_rate,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints"),
        "restore_checkpoint": False,
    }

    os.system("rm -rf output*")
    os.system(f"rm -f characteristic_curve_{simulation}.txt")
    file = open(f"characteristic_curve_{simulation}.txt", "w", buffering=1)  # line-buffered
    file.write("Capillary Pressure,Saturation\n")

    sim = PorousMedia(**kwargs)
    sim.run(args.timesteps)

    file.close()
