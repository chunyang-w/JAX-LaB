"""
Diagnostic: compare WITH and WITHOUT grain wetting (delta_rho_a=0.2 vs 0)
to isolate whether the density spike is caused by contact angle modification.
"""
import sys, os
sys.path.insert(0, "/gpfs/home/cw1722/sponge/JAX-LaB")
os.chdir("/gpfs/home/cw1722/sponge/JAX-LaB")

import numpy as np
import jax.numpy as jnp
import h5py

drho = 0.07; width = 4; buffer = 8
rho_w_l = 2.0; rho_w_g = 0.1; rho_a_l = 2.0; rho_a_g = 0.1

GEOM = "./assets/374_05_03_256.mat"
with h5py.File(GEOM, "r") as f:
    solid_full = np.array(f["bin"], dtype=bool)

Ny = 16; Nz = 16
solid_slice = solid_full[:, :Ny, :Nz]

nx_orig = 256; nx = nx_orig + 2 * buffer; ny = Ny; nz = Nz

from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseBGK
from src.boundary_conditions import BounceBack, EquilibriumBC
from functools import partial, reduce
import operator
from jax import jit, vmap
from jax.tree import map

precision = "f32/f32"
g_kkprime = -0.06 * np.ones((2, 2))
g_kkprime[0, 1] = 0.54; g_kkprime[1, 0] = 0.54

ind_solid = np.where(solid_slice)
idx = np.zeros((len(ind_solid[0]), 3), dtype=int)
idx[:, 0] = ind_solid[0] + buffer; idx[:, 1] = ind_solid[1]; idx[:, 2] = ind_solid[2]

def run_test(wetting_val, n_steps=20, label=""):
    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[tuple(idx.T)] = np.pi / 6
    theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    phi_w = np.ones((nx, ny, nz, 1)); phi_a = np.ones((nx, ny, nz, 1))
    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_a = np.zeros((nx, ny, nz, 1))
    delta_rho_a[tuple(idx.T)] = wetting_val  # test different values

    class PorousMedia(MultiphaseBGK):
        def initialize_macroscopic_fields(self):
            rho_tree = []
            xc = np.arange(self.nx, dtype=float)[:, None, None]
            mid_w = 0.5*(rho_w_l+rho_w_g); half_w = 0.5*(rho_w_l-rho_w_g)
            mid_a = 0.5*(rho_a_l+rho_a_g); half_a = 0.5*(rho_a_l-rho_a_g)
            ic = buffer // 2
            t = np.tanh(2.0*(xc-ic)/width)
            for arr_fn in [lambda t: mid_w+half_w*t, lambda t: mid_a-half_a*t]:
                arr = arr_fn(t) * np.ones((self.nx,self.ny,self.nz))
                r = arr.reshape((self.nx,self.ny,self.nz,1))
                r = self.distributed_array_init((self.nx,self.ny,self.nz,1), self.precisionPolicy.compute_dtype, init_val=r)
                r = self.precisionPolicy.cast_to_output(r)
                rho_tree.append(r)
            u = np.zeros((self.nx,self.ny,self.nz,3))
            u = self.distributed_array_init((self.nx,self.ny,self.nz,3), self.precisionPolicy.compute_dtype, init_val=u)
            u = self.precisionPolicy.cast_to_output(u)
            return rho_tree, [u, u]

        @partial(jit, static_argnums=(0,))
        def compute_potential(self, rho_tree):
            rho_tree = map(lambda r: self.precisionPolicy.cast_to_compute(r), rho_tree)
            U_tree = map(lambda r: jnp.zeros_like(r), rho_tree)
            return rho_tree, U_tree

        @partial(jit, static_argnums=(0,))
        def compute_pressure(self, rho_tree, psi_tree=None):
            return map(lambda r: r * self.lattice.cs2, rho_tree)

        def compute_total_pressure(self, p_tree, rho_tree=None):
            return p_tree[0]+p_tree[1]+3*self.g_kkprime[0,1]*p_tree[1]*p_tree[0]

        @partial(jit, static_argnums=(0,))
        def compute_fluid_fluid_force(self, psi_tree, U_tree):
            c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
            def saf(psi):
                ps = self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q))
                pq = jnp.broadcast_to(psi, ps.shape)
                return ps.at[0:1].set(pq[0:1]).at[-1:].set(pq[-1:])
            ps_t = [saf(p) for p in psi_tree]
            Us_t = [self.streaming(jnp.repeat(U, axis=-1, repeats=self.q)) for U in U_tree]
            def ff1(Ai, gr):
                return reduce(operator.add, [jnp.dot((1-A)*G*self.G_ff*ps,c) for A,G,ps in zip(list(Ai),list(gr),ps_t)])
            def ff2(Ai):
                return reduce(operator.add, [A*jnp.dot(self.G_ff*Us,c) for A,Us in zip(list(Ai),Us_t)])
            return [p*n1+n2 for p,n1,n2 in zip(psi_tree, list(vmap(ff1,in_axes=(0,0))(self.A,self.g_kkprime)), list(vmap(ff2,in_axes=(0,))(self.A)))]

        def set_boundary_conditions(self):
            vel = np.zeros((self.boundingBoxIndices["left"].shape[0],3), dtype=self.precisionPolicy.compute_dtype)
            inlet = self.boundingBoxIndices["left"]
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy,
                (rho_w_g+drho)*np.ones((inlet.shape[0],1), dtype=self.precisionPolicy.compute_dtype), vel))
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy,
                (rho_a_l+drho)*np.ones((inlet.shape[0],1), dtype=self.precisionPolicy.compute_dtype), vel))
            outlet = self.boundingBoxIndices["right"]
            vout = np.zeros((outlet.shape[0],3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy,
                (rho_w_l-drho)*np.ones((outlet.shape[0],1), dtype=self.precisionPolicy.compute_dtype), vout))
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy,
                (rho_a_g-drho)*np.ones((outlet.shape[0],1), dtype=self.precisionPolicy.compute_dtype), vout))
            wall = np.concatenate([self.boundingBoxIndices[d] for d in ["top","bottom","front","back"]])
            for comp in [0, 1]:
                self.BCs[comp].append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))
            wall2 = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall2, self.gridInfo, self.precisionPolicy, theta_w[wall2], phi_w[wall2], delta_rho_w[wall2]))
            self.BCs[1].append(BounceBack(wall2, self.gridInfo, self.precisionPolicy, theta_a[wall2], phi_a[wall2], delta_rho_a[wall2]))

        def output_data(self, **kwargs): pass

    kw = {"n_components": 2, "lattice": LatticeD3Q19(precision),
          "nx": nx, "ny": ny, "nz": nz, "g_kkprime": g_kkprime,
          "body_force": [0.,0.,0.], "omega": [0.6,0.6], "precision": precision,
          "k": [1.,1.], "A": np.zeros((2,2)), "io_rate": -1, "print_info_rate": -1,
          "compute_MLUPS": False, "checkpoint_rate": -1,
          "checkpoint_dir": os.path.abspath("./chk_diag"), "restore_checkpoint": False}
    sim = PorousMedia(**kw)
    f_tree = sim.assign_fields_sharded()
    for t in range(n_steps):
        f_tree, _ = sim.step(f_tree, t)
    rho = [np.array(jnp.sum(f, axis=-1, keepdims=True)) for f in f_tree]
    rw_max = float(rho[0].max())
    rw_nan = int(np.isnan(rho[0]).sum())
    max_idx = np.unravel_index(np.argmax(rho[0]), rho[0].shape)
    print(f"  delta_rho_a={wetting_val:.2f}  rho_w_max={rw_max:.4f}  at x={max_idx[0]}  NaN={rw_nan}")
    return rw_max

print(f"=== CPU test: {nx}x{ny}x{nz} domain ===")
N = 20
for wetting_val in [0.0, 0.1, 0.2]:
    run_test(wetting_val, N)
