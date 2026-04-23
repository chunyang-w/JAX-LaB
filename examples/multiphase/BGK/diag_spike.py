"""
Minimal CPU diagnostic: 1-step test on a thin slice of sphere-pack geometry
to understand the density spike at x=270 after step 0.
"""
import sys
import os
sys.path.insert(0, "/gpfs/home/cw1722/sponge/JAX-LaB")
os.chdir("/gpfs/home/cw1722/sponge/JAX-LaB")

import numpy as np
import jax.numpy as jnp
import h5py

# -------------------------------------------------------------------------
# Parameters (match characteristic_curves.py)
# -------------------------------------------------------------------------
drho    = 0.07
width   = 4
buffer  = 8

rho_w_l = 2.0; rho_w_g = 0.1
rho_a_l = 2.0; rho_a_g = 0.1

# Load real geometry (256^3) but use only a thin slab (x=0..271, y=0..7, z=0..7)
# to keep it fast on CPU
GEOM = "./assets/374_05_03_256.mat"
with h5py.File(GEOM, "r") as f:
    solid_full = np.array(f["bin"], dtype=bool)  # (256,256,256)

# Take a thin slab
Ny = 16; Nz = 16
solid_slice = solid_full[:, :Ny, :Nz]  # (256, Ny, Nz)

nx_orig = 256
nx = nx_orig + 2 * buffer   # 272
ny = Ny; nz = Nz

# -------------------------------------------------------------------------
from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseBGK
from src.boundary_conditions import BounceBack, EquilibriumBC
from functools import partial, reduce
import operator
from jax import jit, vmap
from jax.tree import map

precision = "f32/f32"
g_kkprime = -0.06 * np.ones((2, 2))
g_kkprime[0, 1] = 0.54
g_kkprime[1, 0] = 0.54
simulation = "drainage"

ind_solid = np.where(solid_slice)
idx = np.zeros((len(ind_solid[0]), 3), dtype=int)
idx[:, 0] = ind_solid[0] + buffer
idx[:, 1] = ind_solid[1]
idx[:, 2] = ind_solid[2]

theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
theta_w[tuple(idx.T)] = np.pi / 6
theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))
phi_w = np.ones((nx, ny, nz, 1))
phi_a = np.ones((nx, ny, nz, 1))
delta_rho_w = np.zeros((nx, ny, nz, 1))
delta_rho_a = np.zeros((nx, ny, nz, 1))
delta_rho_a[tuple(idx.T)] = 0.2

class PorousMedia(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        xc = np.arange(self.nx, dtype=float)[:, None, None]
        mid_w = 0.5 * (rho_w_l + rho_w_g)
        half_w = 0.5 * (rho_w_l - rho_w_g)
        mid_a = 0.5 * (rho_a_l + rho_a_g)
        half_a = 0.5 * (rho_a_l - rho_a_g)
        init_center = buffer // 2

        t = np.tanh(2.0 * (xc - init_center) / width)
        rho_arr = (mid_w + half_w * t) * np.ones((self.nx, self.ny, self.nz))
        rho = rho_arr.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        rho_arr = (mid_a - half_a * t) * np.ones((self.nx, self.ny, self.nz))
        rho = rho_arr.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        return [rho_tree[0], rho_tree[1]], [u, u]

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = map(lambda rho: rho * self.lattice.cs2, rho_tree)
        return p_tree

    def compute_total_pressure(self, p_tree, rho_tree=None):
        return p_tree[0] + p_tree[1] + 3 * self.g_kkprime[0, 1] * p_tree[1] * p_tree[0]

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree, U_tree):
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        def stream_and_fix(psi):
            psi_s = self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q))
            psi_q = jnp.broadcast_to(psi, psi_s.shape)
            psi_s = psi_s.at[0:1].set(psi_q[0:1])
            psi_s = psi_s.at[-1:].set(psi_q[-1:])
            return psi_s
        psi_s_tree = [stream_and_fix(psi) for psi in psi_tree]
        U_s_tree = [self.streaming(jnp.repeat(U, axis=-1, repeats=self.q)) for U in U_tree]
        def ffk_1(Ai, g_kkprime_row):
            return reduce(operator.add,
                [jnp.dot((1 - A) * G * self.G_ff * psi_s, c)
                 for A, G, psi_s in zip(list(Ai), list(g_kkprime_row), psi_s_tree)])
        def ffk_2(Ai):
            return reduce(operator.add,
                [A * jnp.dot(self.G_ff * U_s, c) for A, U_s in zip(list(Ai), U_s_tree)])
        return [psi * nt_1 + nt_2
                for psi, nt_1, nt_2 in zip(
                    psi_tree,
                    list(vmap(ffk_1, in_axes=(0, 0))(self.A, self.g_kkprime)),
                    list(vmap(ffk_2, in_axes=(0,))(self.A)))]

    def set_boundary_conditions(self):
        vel = np.zeros((self.boundingBoxIndices["left"].shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
        inlet = self.boundingBoxIndices["left"]
        rho_inlet = (rho_w_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
        rho_inlet = (rho_a_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))

        outlet = self.boundingBoxIndices["right"]
        vel_out = np.zeros((outlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
        rho_outlet = (rho_w_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel_out))
        rho_outlet = (rho_a_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel_out))

        wall = np.concatenate([self.boundingBoxIndices[d] for d in ["top", "bottom", "front", "back"]])
        self.BCs[0].append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))
        wall2 = tuple(idx.T)
        self.BCs[0].append(BounceBack(wall2, self.gridInfo, self.precisionPolicy, theta_w[wall2], phi_w[wall2], delta_rho_w[wall2]))
        wall = np.concatenate([self.boundingBoxIndices[d] for d in ["top", "bottom", "front", "back"]])
        self.BCs[1].append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))
        self.BCs[1].append(BounceBack(wall2, self.gridInfo, self.precisionPolicy, theta_a[wall2], phi_a[wall2], delta_rho_a[wall2]))

    def output_data(self, **kwargs):
        pass  # silence output


kwargs = {
    "n_components": 2,
    "lattice": LatticeD3Q19(precision),
    "nx": nx, "ny": ny, "nz": nz,
    "g_kkprime": g_kkprime,
    "body_force": [0.0, 0.0, 0.0],
    "omega": [0.6, 0.6],
    "precision": precision,
    "k": [1.0, 1.0],
    "A": np.zeros((2, 2)),
    "io_rate": -1,
    "print_info_rate": -1,
    "compute_MLUPS": False,
    "checkpoint_rate": -1,
    "checkpoint_dir": os.path.abspath("./checkpoints_diag"),
    "restore_checkpoint": False,
}

sim = PorousMedia(**kwargs)
f_tree = sim.assign_fields_sharded()

# Print initial rho
def get_rho(f_tree):
    """Sum distributions to get density; shape (nx, ny, nz, 1) -> (nx, ny, nz)."""
    return [np.array(jnp.sum(f, axis=-1, keepdims=True)) for f in f_tree]

rho_init = get_rho(f_tree)
print("=== INITIAL STATE ===")
print(f"rho_w[x=268..271] y=0 z=0: {[float(rho_init[0][x, 0, 0, 0]) for x in range(268, 272)]}")
print(f"rho_a[x=268..271] y=0 z=0: {[float(rho_init[1][x, 0, 0, 0]) for x in range(268, 272)]}")
print(f"rho_w max: {float(rho_init[0].max()):.4f}  min: {float(rho_init[0].min()):.4f}")
print(f"rho_a max: {float(rho_init[1].max()):.4f}  min: {float(rho_init[1].min()):.4f}")

# Run 1 step
f_tree, _ = sim.step(f_tree, 0)

rho1 = get_rho(f_tree)
print("\n=== AFTER STEP 0 ===")
print(f"rho_w[x=268..271] y=0 z=0: {[float(rho1[0][x, 0, 0, 0]) for x in range(268, 272)]}")
print(f"rho_a[x=268..271] y=0 z=0: {[float(rho1[1][x, 0, 0, 0]) for x in range(268, 272)]}")
max_idx = np.unravel_index(np.argmax(rho1[0]), rho1[0].shape)
print(f"rho_w max: {float(rho1[0].max()):.4f}  at {max_idx}")
print(f"rho_a max: {float(rho1[1].max()):.4f}")
print(f"rho_w min: {float(rho1[0].min()):.4f}")
print(f"NaN rho_w: {int(np.isnan(rho1[0]).sum())}")

# Run 5 steps
for t in range(1, 6):
    f_tree, _ = sim.step(f_tree, t)
rho5 = get_rho(f_tree)
print("\n=== AFTER STEP 5 ===")
print(f"rho_w max: {float(rho5[0].max()):.4f}")
print(f"NaN rho_w: {int(np.isnan(rho5[0]).sum())}")
