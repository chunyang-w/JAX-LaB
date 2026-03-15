"""
Soil-water characteristic curve (SWCC) for a 256^3 porous geometry using a
two-component (water + air) Shan-Chen multicomponent multiphase MRT-LBM.

MRT replaces the single BGK relaxation with independently tuned moment-space
relaxation rates. The stress modes that drive divergence during Haines jump
snap-off events (bulk stress, energy flux) are decoupled from the shear
viscosity mode, eliminating the instability seen in the BGK version at ~22000
steps without changing any fluid physics or calibration parameters.

The default spherepack geometry is taken from Digital Rocks Portal (porosity 0.381):
1. https://digitalporousmedia.org/published-datasets/tapis/projects/drp.project.published/drp.project.published.DRP-372/374_05_03/374_05_03_256/

Accepts geometry from either:
  .mat  (HDF5, key "bin", 0=pore 1=solid)  — default sphere pack
  .npy  (bool/int array, True/1=solid)     — e.g. sphere, gdl, blob geometries

Usage:
    python3 characteristic_curves.py                                    # drainage (default)
    python3 characteristic_curves.py --simulation imbibition            # imbibition
    python3 characteristic_curves.py --geometry /path/to/geometry.npy  # custom geometry
"""

import argparse

from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack, EquilibriumBC
from src.utils import save_fields_vtk

import h5py
import phantomgaze as pg
import matplotlib.pyplot as plt

from functools import partial
import os
import numpy as np
import jax.numpy as jnp
from jax import jit, config
from jax.tree import map


# Multi-component droplet simulation to tune fluid-fluid interaction parameters and surface tension
class Droplet3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        dist = (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2 - r**2

        # Water
        rho_inside = rho_w_l
        rho_outside = rho_w_g
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # Air
        rho_inside = rho_a_g
        rho_outside = rho_a_l
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
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

    # NOTE: @partial(jit) intentionally removed from this override. JAX caches the trace
    # based on the number of concrete args at first call. The base run() calls this as
    # compute_total_pressure(p_tree, rho_tree), but if JAX previously compiled a trace
    # from a call with only (p_tree,), it raises "takes 2 positional arguments but 3
    # were given". Removing the decorator lets the base class jit context trace through
    # this function correctly without a separate cached signature.
    def compute_total_pressure(self, p_tree, rho_tree=None):
        p_water = p_tree[0]
        p_air = p_tree[1]
        return p_water + p_air + 3 * self.g_kkprime[0, 1] * p_air * p_water

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        p_water = np.array(kwargs["p_tree"][0][...])
        p_air = np.array(kwargs["p_tree"][1][...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p_water": p_water[..., 0],
            "p_air": p_air[..., 0],
            "rho": rho[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
        }
        offset = 60
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p_air[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        p_south = p_air[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        p_west = p_air[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        p_east = p_air[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        p_back = p_air[self.nx // 2, self.ny // 2, self.nz // 2 - offset, 0]
        p_front = p_air[self.nx // 2, self.ny // 2, self.nz // 2 + offset, 0]
        pressure_difference = p_water[self.nx // 2, self.ny // 2, self.nz // 2, 0] - (p_north + p_south + p_west + p_east + p_front + p_back) / 6
        print(f"Pressure difference for radius = {r}: {pressure_difference}")

        rho_north = rho_water[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_water[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_water[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_g_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_l_pred = rho_water[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error Water Min: {(rho_g_pred - rho_w_g) * 100 / rho_w_g} Max: {(rho_l_pred - rho_w_l) * 100 / rho_w_l}")
        print(f"rho_l: {rho_l_pred}, rho_g: {rho_g_pred}")

        save_fields_vtk(timestep, fields, f"output_{r}", "data")
        if timestep == 20000:
            file.write(f"{r},{pressure_difference}\n")


# Multi-component droplet on wall example to tune contact angle
class DropletOnWall3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        dist = (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2 - r**2

        rho_inside = rho_w_l
        rho_outside = rho_w_g
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 1),
            self.precisionPolicy.compute_dtype,
            init_val=rho,
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # Air
        rho_inside = rho_a_g
        rho_outside = rho_a_l
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        self.BCs[0].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_w[tuple(ind.T)], phi_w[tuple(ind.T)], delta_rho_w[tuple(ind.T)])
        )
        self.BCs[1].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_a[tuple(ind.T)], phi_a[tuple(ind.T)], delta_rho_a[tuple(ind.T)])
        )

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = map(lambda rho: rho * self.lattice.cs2, rho_tree)
        return p_tree

    # NOTE: @partial(jit) intentionally removed — see Droplet3D.compute_total_pressure note.
    def compute_total_pressure(self, p_tree, rho_tree=None):
        p_water = p_tree[0]
        p_air = p_tree[1]
        return p_water + p_air + 3 * self.g_kkprime[0, 1] * p_air * p_water

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "flag": self.solid_mask_streamed[0][..., 0],
        }
        save_fields_vtk(timestep, fields, "output", "data")


class PorousMedia(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        if simulation == "imbibition":
            # Water
            rho = rho_w_g * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
            # Air
            rho = rho_a_l * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
            u = np.zeros((self.nx, self.ny, self.nz, 3))
            u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
            u = self.precisionPolicy.cast_to_output(u)
            u_tree = [u, u]
        else:
            # Water
            rho = rho_w_l * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
            # Air
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

    # NOTE: @partial(jit) intentionally removed — see Droplet3D.compute_total_pressure note.
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
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_a[wall], phi_a[wall], delta_rho_a[wall]))
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
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_a[wall], phi_a[wall], delta_rho_a[wall]))

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

        # Computing capillary pressure, saturation using values inside porous media only
        rho_water = rho_water[buffer : buffer + self.nx, ...]
        rho_air = rho_air[buffer : buffer + self.nx, ...]
        porous = np.array(self.solid_mask_streamed[0][buffer : buffer + self.nx, 1:-1, 1:-1, 0])
        p_water = p_water[buffer : buffer + self.nx, ...]
        p_air = p_air[buffer : buffer + self.nx, ...]
        water = (rho_water > rho_air)[..., 0]
        air = (rho_water < rho_air)[..., 0]
        p_wetting = np.mean(p_water[(~porous & water) == 1])
        p_nonwetting = np.mean(p_air[(~porous & air) == 1])

        P_c = p_nonwetting - p_wetting
        v_w = np.sum((~porous & water))
        v_nw = np.sum((~porous & air))
        S = v_w / (v_w + v_nw)
        file.write(f"{P_c},{S}\n")

        red = pg.SolidColor(color=(1.0, 0.0, 0.0), opacity=1.0)
        grey = pg.SolidColor(color=(0.439, 0.475, 0.757), opacity=0.04)

        dx, dy, dz = (0.01, 0.01, 0.01)
        origin = (0.0, 0.0, 0.0)

        if simulation == "imbibition":
            rho_volume = pg.objects.Volume(
                jnp.array(
                    rho_water[0 : self.nx - 2 * buffer, ..., 0],
                    dtype=self.precisionPolicy.compute_dtype,
                ),
                spacing=(dx, dy, dz),
                origin=origin,
            )
        else:
            rho_volume = pg.objects.Volume(
                jnp.array(
                    rho_air[0 : self.nx - 2 * buffer, ..., 0],
                    dtype=self.precisionPolicy.compute_dtype,
                ),
                spacing=(dx, dy, dz),
                origin=origin,
            )
        boundary_volume = pg.objects.Volume(
            jnp.array(porous[0 : self.nx - 2 * buffer, ...], dtype=jnp.float32),
            spacing=(dx, dy, dz),
            origin=origin,
        )

        radius = 80
        angle = 20 * np.pi / 180
        focal_point = (self.nx * dx / 2, 3 * self.ny * dy / 4, self.nz * dz / 2)
        camera_position = (
            self.nx * dx + radius * np.cos(angle) * dx,
            -0.1,
            -self.nz * dz + radius * np.sin(angle) * dz,
        )

        camera = pg.Camera(
            position=camera_position,
            focal_point=focal_point,
            view_up=(0.0, -1.0, 0.0),
            height=2160,
            width=3840,
            background=pg.SolidBackground(color=(1.0, 1.0, 1.0)),
        )

        rho_threshold = 0.5 * (rho_w_l + rho_w_g)
        screen_buffer = pg.render.contour(rho_volume, threshold=rho_threshold, colormap=red, camera=camera)
        screen_buffer = pg.render.contour(
            boundary_volume,
            camera,
            threshold=0.5,
            colormap=grey,
            screen_buffer=screen_buffer,
        )
        screen_buffer = pg.render.wireframe(
            lower_bound=(0, 0, 0),
            upper_bound=((self.nx - 2 * buffer) * dx, self.ny * dy, self.nz * dz),
            color=pg.SolidColor(color=(0.0, 0.0, 0.0)),
            thickness=0.0025,
            camera=camera,
            screen_buffer=screen_buffer,
        )

        plt.imsave(
            f"images/porous_{simulation}" + str(kwargs["timestep"]).zfill(7) + ".png",
            np.minimum(screen_buffer.image.get(), 1.0),
        )


def _load_geometry(path: str) -> np.ndarray:
    """
    Load a binary porous geometry from .mat or .npy and return a boolean
    array of shape (Gx, Gy, Gz) where True = solid grain.

    .mat  — HDF5 file with dataset "bin"; value 1 = solid.
    .npy  — NumPy array of bool or int; True / 1 = solid.

    Both must be (256, 256, 256) to match the buffer/domain layout.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr.astype(bool)
    elif ext == ".mat":
        with h5py.File(path, "r") as f:
            return np.array(f["bin"], dtype=bool)
    else:
        raise ValueError(f"Unsupported geometry format '{ext}'. Use .npy or .mat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRT SWCC — water/air Shan-Chen LBM")
    parser.add_argument(
        "--geometry",
        default="./assets/374_05_03_256.mat",
        help="Path to geometry file (.npy or .mat, 256³, True/1=solid). Default: sphere-pack .mat",
    )
    parser.add_argument(
        "--simulation",
        default="drainage",
        choices=["drainage", "imbibition"],
        help="Which process to simulate: 'drainage' (default) or 'imbibition'.",
    )
    args = parser.parse_args()
    simulation = args.simulation

    # -------------------------------------------------------------------------
    # RUN_CALIBRATION: set to True to re-run the surface-tension (Droplet3D)
    # and contact-angle (DropletOnWall3D) validation stages.
    #
    # These stages do NOT feed any data into the PorousMedia simulation below —
    # all parameters (g_kkprime, theta_w, delta_rho) are hardcoded from a
    # previous calibration run.  Set this to True only when you need to
    # re-validate those parameter choices from scratch.
    # -------------------------------------------------------------------------
    RUN_CALIBRATION = False

    nx = 256
    ny = 256
    nz = 256

    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    z = np.linspace(0, nz - 1, nz, dtype=int)
    x, y, z = np.meshgrid(x, y, z)

    precision = "f32/f32"
    g_kkprime = -0.06 * np.ones((2, 2))
    g_kkprime[0, 1] = 0.54
    g_kkprime[1, 0] = 0.54

    width = 4  # Liquid vapor interface width

    # Water properties in Lattice units
    rho_w_l = 2.0
    rho_w_g = 0.1
    tau_w = 1.0

    # Air properties (subcritical)
    rho_a_l = 2.0
    rho_a_g = 0.1
    tau_a = 1.0

    A = np.zeros((2, 2))

    # ── MRT collision matrix (D3Q19, Coveney et al. 2002) ────────────────────
    # Transforms populations to moment space where each moment is relaxed
    # independently. s_v controls shear viscosity (= 1/tau); s_e, s_eta, s_q
    # damp the energy and stress modes that drive Haines-jump instabilities in BGK.
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

    # Note: s_rho and s_j must be 0 to conserve mass and momentum exactly.
    # s_v = 1/tau sets shear viscosity, identical to BGK tau=1.0.
    # s_e, s_eta, s_q, s_pi, s_m damp non-hydrodynamic modes — these are the
    # modes that amplify during Haines jumps and cause BGK to diverge.
    # Values from Coveney et al. 2002 (same as displacement_studies.py).
    s_rho = [0.0, 0.0]
    s_e   = [0.8, 0.8]
    s_eta = [0.8, 0.8]
    s_j   = [0.0, 0.0]
    s_q   = [1.1, 1.1]
    s_m   = [1.0, 1.0]
    s_pi  = [1.0, 1.0]
    s_v   = [1 / tau_w, 1 / tau_a]   # = [1.0, 1.0] — same shear viscosity as BGK

    if RUN_CALIBRATION:
        # Stage 1: surface tension calibration via Young-Laplace on isolated droplets.
        os.system("rm -rf output*/")
        file = open("surface_tension.txt", "w")
        file.write("Radius, Pressure Difference\n")
        R = [25, 30, 35, 40]
        for r in R:
            kwargs = {
                "n_components": 2,
                "lattice": LatticeD3Q19(precision),
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "body_force": [0.0, 0.0, 0.0],
                "g_kkprime": g_kkprime,
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
                "A": A,
                "io_rate": 20000,
                "compute_MLUPS": False,
                "print_info_rate": 20000,
                "checkpoint_rate": -1,
                "checkpoint_dir": os.path.abspath("./checkpoints_"),
                "restore_checkpoint": False,
            }
            sim = Droplet3D(**kwargs)
            sim.run(20000)
        file.close()

        # Stage 2: contact angle calibration on a spherical wall.
        R = 30
        r = 25
        sphere = (x - nx / 2) ** 2 + (y - ny / 2) ** 2 + (z - nz / 2 + R + 24) ** 2 - R**2
        ind = np.array(np.where(sphere <= 0)).T

        theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
        theta_w[ind] = np.pi / 6
        phi_w = np.ones((nx, ny, nz, 1))
        theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))
        theta_a[ind] = np.pi - np.pi / 6
        phi_w[ind] = 1.0
        phi_a = np.ones((nx, ny, nz, 1))
        delta_rho_w = np.zeros((nx, ny, nz, 1))
        delta_rho_a = 0.2 * np.ones((nx, ny, nz, 1))

        kwargs = {
            "n_components": 2,
            "lattice": LatticeD3Q19(precision),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "body_force": [0.0, 0.0, 0.0],
            "g_kkprime": g_kkprime,
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
            "A": A,
            "io_rate": 10000,
            "compute_MLUPS": False,
            "print_info_rate": 10000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        sim = DropletOnWall3D(**kwargs)
        sim.run(20000)

    # Stage 3: Pc-S characteristic curves on the 256^3 porous geometry.
    buffer = 8
    solid_mask = _load_geometry(args.geometry)
    assert solid_mask.shape == (256, 256, 256), f"Geometry must be 256x256x256, got {solid_mask.shape}"
    print(f"Geometry loaded from : {args.geometry}")
    print(f"Solid fraction       : {solid_mask.mean():.4f}")

    nx = nx + 2 * buffer
    ind = np.where(solid_mask)
    idx = np.zeros((len(ind[0]), 3), dtype=int)

    # porous geometry — shift grains past the inlet buffer in x
    idx[:, 0] = ind[0] + buffer
    idx[:, 1] = ind[1]
    idx[:, 2] = ind[2]

    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[tuple(idx.T)] = np.pi / 6
    theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))

    phi_w = np.ones((nx, ny, nz, 1))
    phi_w[tuple(idx.T)] = 1.0
    phi_a = np.ones((nx, ny, nz, 1))

    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_a = np.zeros((nx, ny, nz, 1))
    delta_rho_a[tuple(idx.T)] = 0.2

    # Note: drho is the density perturbation applied symmetrically at inlet (+drho) and outlet (-drho),
    # creating a pressure gradient ΔP = 2*drho*cs² that drives Darcy-scale invasion.
    # 0.0092 was the original BGK value but caused divergence at ~22000 steps via Haines jump instability.
    # Halved to 0.0046 to slow the invasion front. MRT further reduces instability risk by independently
    # damping the stress modes that blow up during snap-off events.
    drho = 0.0092

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
        "io_rate": 1000,
        "print_info_rate": 1000,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    os.system("rm -rf output*")
    os.system(f"rm -f characteristic_curve_{simulation}.txt")
    file = open(f"characteristic_curve_{simulation}.txt", "w")
    file.write("Capillary Pressure,Saturation\n")
    sim = PorousMedia(**kwargs)
    sim.run(250000)
    file.close()
