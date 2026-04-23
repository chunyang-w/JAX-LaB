"""
https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html
"""
"""
This script conducts a 2D simulation of flow around a cylinder using the lattice Boltzmann method (LBM). This is a classic problem in fluid dynamics and is often used to examine the behavior of fluid flow over a bluff body.

In this example you'll be introduced to the following concepts:

1. Lattice: A D2Q9 lattice is used, which is a two-dimensional lattice model with nine discrete velocity directions. This type of lattice allows for a precise representation of fluid flow in two dimensions.

2. Boundary Conditions: The script implements several types of boundary conditions:

    BounceBackHalfway: This condition is applied to the cylinder surface, simulating a no-slip condition where the fluid at the cylinder surface has zero velocity.
    ExtrapolationOutflow: This condition is applied at the outlet (right boundary), where the fluid is allowed to exit the simulation domain freely.
    Regularized: This condition is applied at the inlet (left boundary) and models the inflow of fluid into the domain with a specified velocity profile. Another Regularized condition is used for the stationary top and bottom walls.
3. Velocity Profile: The script uses a Poiseuille flow profile for the inlet velocity. This is a parabolic profile commonly seen in pipe flow.

4. Drag and lift calculation: The script computes the lift and drag on the cylinder, which are important quantities in fluid dynamics and aerodynamics.

5. Visualization: The simulation outputs data in VTK format for visualization. It also generates images of the velocity field. The data can be visualized using software like ParaView.

# To run type:
nohup python3 examples/CFD/cylinder2d.py > logfile.log &
"""

import os
import json
import queue
import threading
import jax
from time import time
from jax import config
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils import *
from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.config.update("jax_enable_x64", True)

# config.update("jax_default_matmul_precision", "highest")


def _viz_worker(q: queue.Queue, gif_path: str, plot_path: str):
    """Background thread: renders frames → GIF and updates CL/CD plot."""
    gif_writer = imageio.get_writer(gif_path, mode="I", duration=0.1, loop=0)
    try:
        while True:
            task = q.get()
            if task is None:          # sentinel — time to stop
                break

            # ── GIF frame ──────────────────────────────────────────────────
            if "u_frame" in task:
                u = task["u_frame"]   # (nx, ny, 2) float array
                speed = np.linalg.norm(u, axis=-1).T   # (ny, nx), transposed for imshow
                vmax = float(speed.max()) or 1.0
                fig, ax = plt.subplots(figsize=(11, 2), dpi=80)
                ax.imshow(speed, origin="lower", aspect="auto",
                          cmap="viridis", vmin=0, vmax=vmax)
                ax.axis("off")
                ax.set_title(f"step {task['timestep']}", fontsize=8, pad=2)
                fig.tight_layout(pad=0.2)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                gif_writer.append_data(frame)
                plt.close(fig)

            # ── CL/CD plot ──────────────────────────────────────────────────
            if "cl_hist" in task and len(task["cl_hist"]) > 1:
                t = task["t_hist"]
                cl = task["cl_hist"]
                cd = task["cd_hist"]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                ax1.plot(t, cl, color="steelblue")
                ax1.set_ylabel("CL")
                ax1.grid(True, alpha=0.4)
                ax2.plot(t, cd, color="tomato")
                ax2.set_ylabel("CD")
                ax2.set_xlabel("timestep")
                ax2.grid(True, alpha=0.4)
                fig.tight_layout()
                fig.savefig(plot_path, dpi=100)
                plt.close(fig)

            q.task_done()
    finally:
        gif_writer.close()


class Cylinder(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cl_history: list[float] = []
        self.cd_history: list[float] = []
        self.t_history:  list[int]   = []
        self._viz_queue  = queue.Queue(maxsize=8)   # bounded — keeps memory in check
        self._viz_thread = threading.Thread(
            target=_viz_worker,
            args=(self._viz_queue, "cylinder.gif", "cl_cd_curve.png"),
            daemon=True,
        )
        self._viz_thread.start()

    def _enqueue(self, task: dict):
        """Non-blocking put; silently drops tasks when the worker is behind."""
        try:
            self._viz_queue.put_nowait(task)
        except queue.Full:
            pass

    def finalize_viz(self):
        """Block until the worker has processed everything, then stop it."""
        self._viz_queue.join()           # wait for all queued items to finish
        self._viz_queue.put(None)        # send sentinel
        self._viz_thread.join()

    def set_boundary_conditions(self):
        # Define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2.0 * diam, 2.0 * diam
        cylinder = (xx - cx) ** 2 + (yy - cy) ** 2 <= (diam / 2.0) ** 2
        cylinder = coord[cylinder]
        implicit_distance = np.reshape((xx - cx) ** 2 + (yy - cy) ** 2 - (diam / 2.0) ** 2, (self.nx, self.ny))
        self.BCs.append(InterpolatedBounceBackBouzidi(tuple(cylinder.T), implicit_distance, self.gridInfo, self.precisionPolicy))

        # Outflow BC
        outlet = self.boundingBoxIndices["right"]
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # Inlet BC
        inlet = self.boundingBoxIndices["left"]
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet, yy_inlet.min(), yy_inlet.max() - yy_inlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, "velocity", vel_inlet))

        # No-slip BC for top and bottom
        wall = np.concatenate([self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]])
        vel_wall = np.zeros(wall.shape, dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(Regularized(tuple(wall.T), self.gridInfo, self.precisionPolicy, "velocity", vel_wall))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using bounce-back)
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        if timestep == 0:
            self.CL_max = 0.0
            self.CD_max = 0.0

        # ── GIF frame (every io_rate step) ─────────────────────────────────
        task: dict = {"u_frame": u, "timestep": timestep}

        if timestep > 0.5 * niter_max:
            # compute lift and drag over the cylinder
            cylinder = self.BCs[0]
            boundary_force = cylinder.momentum_exchange_force(kwargs["f_poststreaming"], kwargs["f_postcollision"])
            boundary_force = np.sum(np.array(boundary_force), axis=0)
            drag = boundary_force[0]
            lift = boundary_force[1]
            cd = 2.0 * drag / (prescribed_vel**2 * diam)
            cl = 2.0 * lift / (prescribed_vel**2 * diam)

            u_old = np.linalg.norm(u_prev, axis=2)
            u_new = np.linalg.norm(u, axis=2)
            err = np.sum(np.abs(u_old - u_new))
            self.CL_max = max(self.CL_max, cl)
            self.CD_max = max(self.CD_max, cd)
            print("error= {:07.6f}, CL = {:07.6f}, CD = {:07.6f}".format(err, cl, cd))

            self.cl_history.append(float(cl))
            self.cd_history.append(float(cd))
            self.t_history.append(timestep)
            task["cl_hist"] = list(self.cl_history)
            task["cd_hist"] = list(self.cd_history)
            task["t_hist"]  = list(self.t_history)

        self._enqueue(task)


# Helper function to specify a parabolic poiseuille profile
poiseuille_profile = lambda x, x0, d, umax: np.maximum(0.0, 4.0 * umax / (d**2) * ((x - x0) * d - (x - x0) ** 2))

if __name__ == "__main__":
    precision = "f64/f64"
    # diam_list = [10, 20, 30, 40, 60, 80]
    diam_list = [80]
    CL_list, CD_list = [], []
    result_dict = {}
    result_dict["resolution_list"] = diam_list
    for diam in diam_list:
        scale_factor = 80 / diam
        prescribed_vel = 0.003 * scale_factor
        lattice = LatticeD2Q9(precision)

        nx = int(22 * diam)
        ny = int(4.1 * diam)

        Re = 100.0
        visc = prescribed_vel * diam / Re
        omega = 1.0 / (3.0 * visc + 0.5)

        os.system("rm -rf ./*.vtk && rm -rf ./*.png")

        kwargs = {
            "lattice": lattice,
            "omega": omega,
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "precision": precision,
            "io_rate": int(500 / scale_factor),
            "print_info_rate": int(10000 / scale_factor),
            "return_fpost": True,  # Need to retain fpost-collision for computation of lift and drag
        }
        # characteristic time
        tc = prescribed_vel / diam
        niter_max = int(100 // tc)
        sim = Cylinder(**kwargs)
        sim.run(niter_max)
        sim.finalize_viz()   # flush queue, close GIF writer
        CL_list.append(sim.CL_max)
        CD_list.append(sim.CD_max)

    result_dict["CL"] = CL_list
    result_dict["CD"] = CD_list
    with open("data.json", "w") as fp:
        json.dump(result_dict, fp)
