r"""
Direction-aware 3D cone filtration (the "TDA" descriptor family).

For a probing direction d, every solid voxel receives three filtration values;
pore voxels receive the sentinel 1.25 in every channel, so they enter each
sublevel set only after all solid voxels. The construction follows Ref.
[Topolnicki et al. 2026].

Channels
--------
0. cone      1 - (solid voxels inside the cone) / (voxels in the cone).
             The double cone has its apex on the voxel and opens along +-d,
             with base radius `--cone-radius` at half-height `--cone-height`:
             an offset o is inside when |o.d| <= height and
             |o - (o.d) d| <= radius |o.d| / height. 0 means the cone is all
             solid, near 1 means it is open. This channel carries directional
             connectivity and is the only one used for persistent homology.
1. dir_long  1 - |v1.d|, v1 the leading eigenvector of the covariance of the
             solid voxels in a sphere of radius `--radius` (PCA). Small means
             the local solid is elongated along d.
2. dir_norm  1 - |v3.d|, v3 the smallest eigenvector, i.e. the local plate
             normal. Small means d pierces a plate. In 3D the identity
             sum_i |v_i.d|^2 = 1 leaves two independent projections, so both
             channels carry information.

Channels 0-2 together are the input of the Euler characteristic profile
(`ecp3d.jl`).

Implementation notes
--------------------
* No grid rotation. The cone offsets are built directly along d and the grid
  is indexed modulo its shape. Rotating the grid to align d with an axis would
  require interpolation for every diagonal and would fill the rotated-in
  corners with pore; the structures are periodic on the 3-torus (the domain
  the LBM solves), so periodic indexing is exact and treats all nine
  directions identically.
* The PCA channels are frame-free projections onto d, so no transverse
  reference axis is needed; the 3x3 eigenproblem is solved exactly with
  `np.linalg.eigh`.

Frame
-----
Arrays are indexed [z, y, x] = [solver_z, solver_y, solver_x], as returned by
`structure_io3d.load_structure`. `--direction H K L` is (dx, dy, dz) in the
solver frame (the frame of `structures.csv`'s e1_x/e1_y/e1_z and of the
permeability tensor). `compute()` converts once; `compute_arr()` takes the
direction already in array order. Do not transpose the array.

Input / output
--------------
Reads `<dataset>/structures.csv` and `<dataset>/structures/<filename>.raw`;
writes one `<filename>.npy` per structure to `--outputdir`, a float32 array of
shape (Z, Y, X, C) with C = `--channels`. Existing outputs are skipped.

Usage
-----
    python scripts/filtration3d.py --self-test
    python scripts/filtration3d.py --dataset DATA/aniso3d \
        --outputdir DESC/aniso3d/filt/a1_-1_0 --direction 1 -1 0 \
        [--radius 4] [--cone-radius 3] [--cone-height 6] [--workers 24] [--limit N]

`--direction 1 -1 0` parses correctly because no option string of this parser
looks like a negative number; do not add a short option such as -1.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io3d import load_structure          # noqa: E402

VOID = np.float32(1.25)

# Below this many solid voxels in the sphere the 3x3 covariance is rank
# deficient and v3 is arbitrary, so both PCA channels are set to 1.0. With
# radius 4 the sphere holds 257 voxels, so this fires only for isolated voxels.
MIN_PCA_NEIGHBOURS = 4


# ---------------------------------------------------------------------------
# Offset stencils   (all offsets are in array order: (oz, oy, ox))
# ---------------------------------------------------------------------------

def cone_offsets(d_arr, radius: float, height: float) -> np.ndarray:
    """(oz, oy, ox) offsets inside the double cone pointing along +-d_arr."""
    d = np.asarray(d_arr, dtype=np.float64)
    n = np.linalg.norm(d)
    if n == 0.0:
        raise ValueError("direction must be non-zero")
    d = d / n
    lim = int(np.ceil(height + radius)) + 1

    # The tolerance is required: floating-point round-off (e.g. a projection of
    # 1e-17 instead of 0) would otherwise flip offsets lying exactly on the cone
    # boundary in or out, giving symmetry-equivalent directions cones of
    # different size and breaking central symmetry.
    #
    # The stencil is equal-sized within a symmetry class (the three axes agree
    # exactly, and so do the six face diagonals), but an axis cone holds more
    # voxels than a diagonal one (156 vs 116 at radius 3, height 6). This is a
    # property of the cubic lattice and does not bias the channel, which is
    # normalised by its own offset count. The self-test checks equality within
    # each class.
    EPS = 1e-9

    offs = []
    for oz in range(-lim, lim + 1):
        for oy in range(-lim, lim + 1):
            for ox in range(-lim, lim + 1):
                if oz == 0 and oy == 0 and ox == 0:
                    continue
                v = np.array([oz, oy, ox], dtype=np.float64)
                along = float(v @ d)
                perp = float(np.linalg.norm(v - along * d))
                if (abs(along) <= height + EPS
                        and perp <= radius * abs(along) / height + EPS):
                    offs.append((oz, oy, ox))
    return np.asarray(offs, dtype=np.int16).reshape(-1, 3)


def sphere_offsets(radius: int) -> np.ndarray:
    """(oz, oy, ox) offsets with oz^2 + oy^2 + ox^2 <= radius^2."""
    r2 = radius * radius
    offs = [(oz, oy, ox)
            for oz in range(-radius, radius + 1)
            for oy in range(-radius, radius + 1)
            for ox in range(-radius, radius + 1)
            if oz * oz + oy * oy + ox * ox <= r2]
    return np.asarray(offs, dtype=np.int16).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@njit(inline="always")
def _at(grid, k, j, i):
    return grid[k % grid.shape[0], j % grid.shape[1], i % grid.shape[2]]


@njit(parallel=True, cache=True)
def _filtration(grid, cone_off, sph_off, d0, d1, d2):
    """grid: uint8 (Z, Y, X), 1 = solid. Returns (Z, Y, X, 3) float32."""
    Z, Y, X = grid.shape
    out = np.empty((Z, Y, X, 3), dtype=np.float32)
    n_cone = cone_off.shape[0]
    n_sph = sph_off.shape[0]

    for k in prange(Z):
        for j in range(Y):
            for i in range(X):
                if grid[k, j, i] == 0:
                    out[k, j, i, 0] = 1.25
                    out[k, j, i, 1] = 1.25
                    out[k, j, i, 2] = 1.25
                    continue

                # --- channel 0: cone occupancy ----------------------------
                filled = 0
                for m in range(n_cone):
                    if _at(grid, k + cone_off[m, 0], j + cone_off[m, 1],
                           i + cone_off[m, 2]) != 0:
                        filled += 1
                out[k, j, i, 0] = 1.0 - filled / n_cone if n_cone > 0 else 1.0

                # --- channels 1-2: local orientation vs d -----------------
                n = 0.0
                s0 = s1 = s2 = 0.0
                c00 = c01 = c02 = c11 = c12 = c22 = 0.0
                for m in range(n_sph):
                    a = sph_off[m, 0]
                    b = sph_off[m, 1]
                    c = sph_off[m, 2]
                    if _at(grid, k + a, j + b, i + c) == 0:
                        continue
                    n += 1.0
                    fa = float(a)
                    fb = float(b)
                    fc = float(c)
                    s0 += fa
                    s1 += fb
                    s2 += fc
                    c00 += fa * fa
                    c01 += fa * fb
                    c02 += fa * fc
                    c11 += fb * fb
                    c12 += fb * fc
                    c22 += fc * fc

                if n < MIN_PCA_NEIGHBOURS:
                    out[k, j, i, 1] = 1.0
                    out[k, j, i, 2] = 1.0
                    continue

                m0 = s0 / n
                m1 = s1 / n
                m2 = s2 / n
                C = np.empty((3, 3), dtype=np.float64)
                C[0, 0] = c00 / n - m0 * m0
                C[0, 1] = c01 / n - m0 * m1
                C[0, 2] = c02 / n - m0 * m2
                C[1, 0] = C[0, 1]
                C[1, 1] = c11 / n - m1 * m1
                C[1, 2] = c12 / n - m1 * m2
                C[2, 0] = C[0, 2]
                C[2, 1] = C[1, 2]
                C[2, 2] = c22 / n - m2 * m2

                # eigh returns eigenvalues ascending: v1 = V[:, 2], v3 = V[:, 0]
                w, V = np.linalg.eigh(C)
                out[k, j, i, 1] = 1.0 - abs(V[0, 2] * d0 + V[1, 2] * d1
                                            + V[2, 2] * d2)
                out[k, j, i, 2] = 1.0 - abs(V[0, 0] * d0 + V[1, 0] * d1
                                            + V[2, 0] * d2)
    return out


def compute_arr(grid, d_arr, radius=4, cone_radius=3.0, cone_height=6.0,
                channels=3):
    """Filtration for a direction already in array order (dz, dy, dx)."""
    d = np.asarray(d_arr, dtype=np.float64)
    d = d / np.linalg.norm(d)
    out = _filtration(np.ascontiguousarray(grid, dtype=np.uint8),
                      cone_offsets(d, cone_radius, cone_height),
                      sphere_offsets(radius),
                      float(d[0]), float(d[1]), float(d[2]))
    return out if channels == 3 else np.ascontiguousarray(out[..., :channels])


def compute(grid, direction, radius=4, cone_radius=3.0, cone_height=6.0,
            channels=3):
    """Filtration of a binary structure (1 = solid) along (dx, dy, dz)."""
    dx, dy, dz = direction
    return compute_arr(grid, (dz, dy, dx), radius, cone_radius, cone_height,
                       channels)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _one(args):
    path, out_path, direction, radius, cr, ch, channels = args
    if os.path.exists(out_path):
        return "cached"
    # One numba thread per worker: the pool already saturates the machine, and
    # nesting prange inside N processes oversubscribes badly.
    import numba
    numba.set_num_threads(1)
    filt = compute(load_structure(path), direction, radius, cr, ch, channels)
    np.save(out_path, filt)
    return "ok"


def run(dataset, outputdir, direction, radius, cr, ch, channels, workers, limit,
        start=0):
    """
    Filtrations for rows [start, start + limit) of structures.csv.

    One direction over 5000 structures is about 31 GB of float32 at 80^3 with
    three channels, so large runs process the dataset in chunks (compute the
    filtrations, reduce them to descriptors, delete them) to bound peak disk use.
    """
    import pandas as pd
    df = pd.read_csv(os.path.join(dataset, "structures.csv"))
    end = start + limit if limit else len(df)
    df = df.iloc[start:end]
    os.makedirs(outputdir, exist_ok=True)

    tasks = [(os.path.join(dataset, "structures", r.filename),
              os.path.join(outputdir, r.filename.replace(".raw", ".npy")),
              direction, radius, cr, ch, channels)
             for r in df.itertuples()]

    from tqdm import tqdm
    done = 0
    tag = "[{:g} {:g} {:g}] {}..{}".format(*direction, start, end)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"filtration {tag}"):
            done += f.result() == "ok"
    print(f"  {done} computed, {len(tasks) - done} cached -> {outputdir}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # NOTE: cone_offsets() takes the direction in ARRAY order (dz, dy, dx),
    # so these are labelled by what they mean in the solver frame.
    DIRS = {"x": (0, 0, 1), "y": (0, 1, 0), "z": (1, 0, 0),
            "xy": (0, 1, 1), "x-y": (0, -1, 1), "yz": (1, 1, 0)}

    # --- stencil geometry --------------------------------------------------
    for name, d in DIRS.items():
        c = cone_offsets(d, 3.0, 6.0)
        sym = {tuple(o) for o in c} == {tuple(-o) for o in c}
        check(f"cone d={name} is centrally symmetric (double-ended)", sym,
              f"{len(c)} offsets")

    n = {k: len(cone_offsets(d, 3.0, 6.0)) for k, d in DIRS.items()}
    check("cone size invariant under axis permutation of d",
          n["x"] == n["y"] == n["z"], f"x={n['x']} y={n['y']} z={n['z']}")
    check("cone size invariant under sign flip of a component",
          n["xy"] == n["x-y"], f"{n['xy']} vs {n['x-y']}")
    check("cone size invariant across the face diagonals",
          n["xy"] == n["yz"], f"xy={n['xy']} yz={n['yz']}")

    c0 = cone_offsets(DIRS["x"], 3.0, 6.0)          # array order (oz, oy, ox)
    check("d=x cone extends along x, not y or z",
          np.abs(c0[:, 2]).max() > max(np.abs(c0[:, 0]).max(),
                                       np.abs(c0[:, 1]).max()),
          f"|ox|max={np.abs(c0[:,2]).max()} |oy|max={np.abs(c0[:,1]).max()} "
          f"|oz|max={np.abs(c0[:,0]).max()}")

    # --- sentinels ---------------------------------------------------------
    solid = np.ones((24, 24, 24), dtype=np.uint8)
    f = compute(solid, (1, 0, 0))
    check("all-solid gives cone = 0", np.allclose(f[..., 0], 0.0),
          f"max {f[..., 0].max():.3g}")

    g = np.ones((24, 24, 24), dtype=np.uint8)
    g[12, 12, 12] = 0
    f = compute(g, (1, 0, 0))
    check("void voxel gets the 1.25 sentinel in all 3 channels",
          np.all(f[12, 12, 12] == 1.25))

    # --- direction sensitivity --------------------------------------------
    # Rods running along solver x: solid for all x at a lattice of (y, z).
    N = 48
    rods = np.zeros((N, N, N), dtype=np.uint8)
    for z0 in range(0, N, 16):
        for y0 in range(0, N, 16):
            rods[z0:z0 + 5, y0:y0 + 5, :] = 1
    fx = compute(rods, (1, 0, 0))
    fy = compute(rods, (0, 1, 0))
    m_along = fx[..., 0][rods == 1].mean()
    m_across = fy[..., 0][rods == 1].mean()
    check("cone is direction-sensitive on rods", m_along < m_across - 0.3,
          f"mean cone along={m_along:.3f} across={m_across:.3f}")
    check("dir_long ~ 0 along a rod's axis, ~ 1 across",
          fx[..., 1][rods == 1].mean() < 0.1 < 0.9 < fy[..., 1][rods == 1].mean(),
          f"dir_long along={fx[...,1][rods==1].mean():.3f} "
          f"across={fy[...,1][rods==1].mean():.3f}")

    # Plates normal to solver x: solid slabs at a lattice of x, full in y and z.
    plates = np.zeros((N, N, N), dtype=np.uint8)
    for x0 in range(0, N, 16):
        plates[:, :, x0:x0 + 3] = 1
    px = compute(plates, (1, 0, 0))
    check("dir_norm ~ 0 along a plate's normal",
          px[..., 2][plates == 1].mean() < 0.1,
          f"dir_norm={px[...,2][plates==1].mean():.3f} "
          f"dir_long={px[...,1][plates==1].mean():.3f}")

    # --- exact axis-permutation equivariance ------------------------------
    # Relabelling grid axes is lossless (no voxel changes value), so permuting
    # the grid and the direction together must permute the filtration exactly.
    # A [z,y,x] vs [x,y,z] slip in the offset construction passes every other
    # check on this list and fails here.
    rng = np.random.default_rng(0)
    s = (rng.random((20, 20, 20)) > 0.6).astype(np.uint8)
    d_arr = np.array([0.0, 1.0, -1.0])              # array order (dz, dy, dx)
    ref = compute_arr(s, d_arr)
    worst = 0.0
    for t in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        s_p = np.ascontiguousarray(np.transpose(s, t))
        got = compute_arr(s_p, d_arr[list(t)])
        want = np.transpose(ref, t + (3,))
        worst = max(worst, float(np.abs(got - want).max()))
    check("axis permutation of (grid, d) permutes the filtration exactly",
          worst == 0.0, f"max diff {worst:.2e}")

    # --- periodicity -------------------------------------------------------
    sh = (3, -5, 7)
    rolled = np.ascontiguousarray(np.roll(s, sh, axis=(0, 1, 2)))
    fr = compute_arr(rolled, d_arr)
    want = np.roll(ref, sh, axis=(0, 1, 2))
    check("cyclic shift of the structure shifts the filtration exactly",
          np.abs(fr - want).max() == 0.0,
          f"max diff {np.abs(fr - want).max():.2e}")

    # --- channel count -----------------------------------------------------
    check("--channels 2 drops dir_norm and keeps the rest",
          np.array_equal(compute(g, (1, 0, 0), channels=2), f[..., :2]))
    check("--channels 1 keeps the cone channel alone",
          np.array_equal(compute(g, (1, 0, 0), channels=1), f[..., :1]))

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--outputdir")
    p.add_argument("--direction", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   metavar=("H", "K", "L"),
                   help="(dx, dy, dz) in the solver frame")
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--cone-radius", type=float, default=3.0)
    p.add_argument("--cone-height", type=float, default=6.0)
    p.add_argument("--channels", type=int, default=3, choices=(1, 2, 3),
                   help="number of channels to write (cone, dir_long, "
                        "dir_norm). The kernel always computes all three; this "
                        "only trims the file. 1 suffices for persistent "
                        "homology (ph3d.py reads channel 0 only) and gives "
                        "2.05 MB per 80^3 structure instead of 6.1 MB; ECP "
                        "needs all 3.")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0,
                   help="number of structures to process (0 = all)")
    p.add_argument("--start", type=int, default=0,
                   help="index of the first structure, for chunked runs")
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.outputdir):
        p.error("--dataset and --outputdir are required (or use --self-test)")
    run(a.dataset, a.outputdir, tuple(a.direction), a.radius, a.cone_radius,
        a.cone_height, a.channels, a.workers, a.limit, a.start)


if __name__ == "__main__":
    main()
