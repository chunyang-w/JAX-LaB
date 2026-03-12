"""
analyse_geometry.py — Porous geometry analysis tool for JAX-LaB simulations.

Accepts .npy (bool) or .mat (HDF5, key "bin") geometry files and prints a
structured summary covering: shape, solid/pore fractions, connected-component
topology, pore-size distribution, and a mid-domain cross-section text map.

Usage:
    python3 scripts/analyse_geometry.py path/to/geometry.npy
    python3 scripts/analyse_geometry.py path/to/geometry.mat
    python3 scripts/analyse_geometry.py geom1.npy geom2.npy   # compare
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import label, distance_transform_edt


def load(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(bool)
    elif ext == ".mat":
        with h5py.File(path, "r") as f:
            return np.array(f["bin"], dtype=bool)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .npy or .mat")


def analyse(path: str) -> dict:
    name = Path(path).stem
    a = load(path)
    Gx, Gy, Gz = a.shape

    solid = a        # True  = solid grain
    pore  = ~a       # False = pore space

    # Connected components
    lab_s, n_s = label(solid)
    lab_p, n_p = label(pore)
    sizes_s = np.bincount(lab_s.ravel())[1:] if n_s else np.array([0])
    sizes_p = np.bincount(lab_p.ravel())[1:] if n_p else np.array([0])

    # Pore-size distribution via distance transform (local inscribed sphere radius)
    dt         = distance_transform_edt(pore)
    pore_radii = dt[pore]

    return dict(
        name=name, path=path, shape=a.shape,
        solid_frac=float(solid.mean()), pore_frac=float(pore.mean()),
        solid_vox=int(solid.sum()), pore_vox=int(pore.sum()),
        n_solid_comp=n_s, n_pore_comp=n_p,
        solid_largest=int(sizes_s.max()), solid_smallest=int(sizes_s.min()),
        pore_largest=int(sizes_p.max()),  pore_smallest=int(sizes_p.min()),
        pore_r_mean=float(pore_radii.mean()),
        pore_r_max=float(pore_radii.max()),
        pore_r_std=float(pore_radii.std()),
        arr=a,
    )


def print_report(r: dict, slice_size: int = 24):
    sep = "─" * 70
    print(sep)
    print(f"  {r['name']}   ({r['path']})")
    print(sep)
    print(f"  Shape         : {r['shape']}")
    print(f"  Solid (True)  : {r['solid_frac']:.4f}  ({r['solid_vox']:>12,} vox)")
    print(f"  Pore  (False) : {r['pore_frac']:.4f}  ({r['pore_vox']:>12,} vox)")
    print()
    print(f"  Solid components : {r['n_solid_comp']:>5}  "
          f"largest {r['solid_largest']:>12,}  smallest {r['solid_smallest']:>8,}")
    print(f"  Pore  components : {r['n_pore_comp']:>5}  "
          f"largest {r['pore_largest']:>12,}  smallest {r['pore_smallest']:>8,}")
    print()
    print(f"  Pore-size distribution (inscribed sphere radius, lattice units):")
    print(f"    mean = {r['pore_r_mean']:.2f}   std = {r['pore_r_std']:.2f}   max = {r['pore_r_max']:.2f}")
    print()

    # Mid-domain text map
    a  = r['arr']
    cx = a.shape[0] // 2
    cy = a.shape[1] // 2
    cz = a.shape[2] // 2
    h  = slice_size // 2
    sl = a[cx, cy - h:cy + h, cz - h:cz + h]
    print(f"  Mid-slice (x={cx}, y={cy-h}:{cy+h}, z={cz-h}:{cz+h})  X=solid .=pore")
    for row in sl:
        print("    " + "".join("X" if v else "." for v in row))
    print()


def main():
    parser = argparse.ArgumentParser(description="Porous geometry analyser")
    parser.add_argument("paths", nargs="+", help=".npy or .mat geometry files")
    parser.add_argument("--slice-size", type=int, default=24,
                        help="Side length of the text-map cross-section (default 24)")
    args = parser.parse_args()

    for path in args.paths:
        r = analyse(path)
        print_report(r, slice_size=args.slice_size)


if __name__ == "__main__":
    main()
