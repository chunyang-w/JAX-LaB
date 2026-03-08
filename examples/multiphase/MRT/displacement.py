"""
CO2 drainage simulation in a Berea sandstone porous medium.

Runs only the large-scale PorousMedia (third) stage of displacement_studies.py.
Parameters (fluid densities, contact angle, wettability) are the calibrated
values determined from the Droplet3D and DropletOnWall3D calibration runs.

Geometry source:
    E. Santos, Javier, et al. "3D Dataset of Simulations." Digital Rocks Portal.
    https://www.doi.org/10.17612/93pd-y471
"""

import os
import numpy as np
import h5py

from src.lattice import LatticeD3Q19
from src.eos import Peng_Robinson
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk


class PorousMedia(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        # Water — fills pore space initially
        rho = rho_w_l * np.ones((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # CO2 — injected from the left buffer region
        rho = rho_c_g * np.ones((self.nx, self.ny, self.nz, 1))
        rho[0:buffer, ...] = rho_c_l
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        wall = tuple(idx.T)
        self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
        self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_c[wall], phi_c[wall], delta_rho_c[wall]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, 1:-1, 1:-1, 1:-1, :])
        p_water = np.array(kwargs.get("p_tree")[0][1:-1, 1:-1, 1:-1])
        p_CO2 = np.array(kwargs.get("p_tree")[1][1:-1, 1:-1, 1:-1])
        u = np.array(kwargs.get("u_total")[0, 1:-1, 1:-1, 1:-1, :])
        rho_water = np.array(kwargs.get("rho_tree")[0][0, 1:-1, 1:-1, 1:-1, :])
        rho_CO2 = np.array(kwargs.get("rho_tree")[1][0, 1:-1, 1:-1, 1:-1, :])
        u_water = np.array(kwargs["u_tree"][0][0, 1:-1, 1:-1, 1:-1, :])
        u_CO2 = np.array(kwargs["u_tree"][1][0, 1:-1, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "ux_CO2": u_CO2[..., 0],
            "uy_CO2": u_CO2[..., 1],
            "uz_CO2": u_CO2[..., 2],
            "ux_water": u_water[..., 0],
            "uy_water": u_water[..., 1],
            "uz_water": u_water[..., 2],
            "rho_water": rho_water[..., 0],
            "rho_CO2": rho_CO2[..., 0],
            "p_CO2": p_CO2[..., 0],
            "p_water": p_water[..., 0],
            "phi_w": phi_w[1:-1, 1:-1, 1:-1, 0],
            "theta_c": theta_c[1:-1, 1:-1, 1:-1, 0],
            "flag": self.solid_mask_streamed[0][1:-1, 1:-1, 1:-1, 0],
        }
        save_fields_vtk(timestep, fields, "output", "data")


if __name__ == "__main__":
    precision = "f32/f32"

    # ── Fluid-fluid interaction ───────────────────────────────────────────────
    g_kkprime = -1 * np.ones((2, 2))
    g = 0.00323
    g_kkprime[0, 1] = g
    g_kkprime[1, 0] = g

    # ── Calibrated co-existence densities (from Droplet3D calibration) ────────
    rho_w_l = 8.105   # water liquid
    rho_w_g = 0.016   # water vapour (solubility in CO2)
    rho_c_l = 2.562   # CO2 liquid (dissolved in water)
    rho_c_g = 0.387   # CO2 gas

    # ── Relaxation times ──────────────────────────────────────────────────────
    tau_w = 1.43
    tau_c = 1.0

    # ── MRT collision matrix (D3Q19, Coveney et al. 2002) ────────────────────
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

    s_rho = [0.0, 0.0]
    s_e   = [0.8, 0.8]
    s_eta = [0.8, 0.8]
    s_j   = [0.0, 0.0]
    s_q   = [1.1, 1.1]
    s_m   = [1.0, 1.0]
    s_pi  = [1.0, 1.0]
    s_v   = [1 / tau_w, 1 / tau_c]

    # ── Peng-Robinson EOS ─────────────────────────────────────────────────────
    Tc_w = 0.03646 * 473.15 / 647.1   # 200°C in lattice units
    a        = [1 / 49,  0.01348]
    b        = [2 / 21,  0.13385]
    R        = [1,       1]
    pr_omega = [0.344,   0.22491]
    eos = Peng_Robinson(a=a, b=b, pr_omega=pr_omega, R=R, T=Tc_w)

    # ── Domain ────────────────────────────────────────────────────────────────
    width  = 10
    buffer = 52
    nx = 256 + buffer + 4
    ny = 256
    nz = 256

    # ── Geometry: Berea sandstone (Digital Rocks Portal) ─────────────────────
    geometry = h5py.File("./assets/374_09_03_256.mat", "r")
    _bin = np.array(geometry["bin"], dtype=int)
    ind  = np.where(_bin == 1.0)
    idx  = np.zeros((len(ind[0]), 3), dtype=int)
    idx[:, 0] = ind[0] + buffer
    idx[:, 1] = ind[1]
    idx[:, 2] = ind[2]

    # ── Wettability (calibrated from DropletOnWall3D) ─────────────────────────
    # Water: water-wet rock (contact angle π/6 on solid)
    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[tuple(idx.T)] = np.pi / 6
    phi_w = np.ones((nx, ny, nz, 1))
    phi_w[tuple(idx.T)] = 1.17
    delta_rho_w = np.zeros((nx, ny, nz, 1))

    # CO2: neutral wetting on solid
    theta_c = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    phi_c = np.ones((nx, ny, nz, 1))
    delta_rho_c = np.zeros((nx, ny, nz, 1))

    # ── Simulation ────────────────────────────────────────────────────────────
    kwargs = {
        "n_components": 2,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        "body_force": [1e-4, 0.0, 0.0],
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
        "k": [0.125, 0.125],
        "EOS": eos,
        "A": np.zeros((2, 2)),
        "io_rate": 1000,
        "compute_MLUPS": False,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    os.system("rm -rf output*")
    sim = PorousMedia(**kwargs)
    sim.run(30000)
