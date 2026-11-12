r"""
Persistent homology of a 3D filtration, vectorised as persistence images.

For every filtration `.npy` in `--inputdir`, builds a periodic cubical complex
(gudhi `PeriodicCubicalComplex`, periodic in all three axes because the
structures live on the 3-torus the LBM solves), computes persistence in
dimensions 0, 1 and 2, drops the essential (infinite) bars, and turns each
diagram into a persistence image. The input is either a cone filtration from
`filtration3d.py` (4D array; only channel 0, the cone channel, is used) or a
throat filtration from `filtration_throat3d.py` (3D array, used as is). The
construction follows Ref. [Topolnicki et al. 2026].

Persistence images
------------------
* Weight: `--weight linear` (default) weights each bar by its persistence, so
  the many short near-diagonal bars of a porous structure do not dominate.
  gudhi applies `BirthPersistenceTransform` before calling the weight, so the
  weight function receives (birth, persistence) and the linear weight is x[1].
* Range and bandwidth are set per dimension, because H0, H1 and H2 live on
  different scales: `--im_range`/`--bandwidth` give shared values and
  `--im_range_h{0,1,2}`/`--bandwidth_h{0,1,2}` override them. The Gaussian
  bandwidth is in persistence units, not pixels, and must be comparable to a
  pixel of the chosen range and resolution. gudhi's default of 1.0 is about
  eight pixels at range [0, 1.25] and resolution 10, which blurs every diagram
  into the same blob and leaves essentially one degree of freedom per image.
  A per-dimension range therefore needs a matching per-dimension bandwidth.
  At startup the script prints the bandwidth in pixels per dimension and
  flags a kernel wider than a third of the shorter image extent as SMEARED.

Output
------
One `<name>.npy` per input file in `--outputdir`: the three flattened images
concatenated, H0 then H1 then H2, i.e. 3 x 10 x 10 = 300 values at the default
resolution. A dimension with no finite bars gives zeros. Existing outputs are
skipped. `collect_descriptors.py` assembles the per-direction directories into
columns `ph_<tag>_<i>`.

Cost: about 4 s and 420 MB peak memory per 80^3 structure, the most expensive
stage of the descriptor pipeline.

Usage
-----
Cone filtration (settings used for the paper):

    python scripts/ph3d.py --inputdir DESC/aniso3d/filt/a1_1_0 \
        --outputdir DESC/aniso3d/ph/a1_1_0 --weight linear --resolution 10 10 \
        --im_range_h0 0 1.25 0 1.25  --bandwidth_h0 0.125 \
        --im_range_h1 0 1.25 0 0.80  --bandwidth_h1 0.100 \
        --im_range_h2 0.2 1.0 0 0.06 --bandwidth_h2 0.019 [--workers 24]

Throat filtration (--mode dir, R_cap = 10):

    python scripts/ph3d.py --inputdir DESC/aniso3d_throat/filt/a1_1_0 \
        --outputdir DESC/aniso3d_throat/ph/a1_1_0 --weight linear \
        --im_range_h0 0 0.9 0 0.40 --bandwidth_h0 0.060 \
        --im_range_h1 0 0.9 0 0.75 --bandwidth_h1 0.082 \
        --im_range_h2 0 0.9 0 0.50 --bandwidth_h2 0.067
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gudhi as gd
from gudhi.representations import DiagramSelector, PersistenceImage
from tqdm import tqdm

DIMS = (0, 1, 2)

# Built inside the worker: a lambda would not survive being pickled to the pool.
# The argument is (birth, persistence); see the module docstring.
WEIGHTS = {
    "constant":  lambda x: 1.0,
    "linear":    lambda x: x[1],
    "quadratic": lambda x: x[1] * x[1],
}


def _one(args):
    fname, inputdir, outputdir, resolution, params, weight = args
    out = os.path.join(outputdir, fname)
    if os.path.exists(out):
        return "cached"

    filt = np.load(os.path.join(inputdir, fname))
    if filt.ndim == 4:
        filt = filt[:, :, :, 0]                    # cone channel

    cx = gd.PeriodicCubicalComplex(
        top_dimensional_cells=filt.astype(np.float64),
        periodic_dimensions=[True, True, True])
    cx.compute_persistence()

    sel = DiagramSelector(use=True, limit=np.inf, point_type="finite")

    vecs = []
    for d in DIMS:
        bandwidth, im_range = params[d]
        bars = sel(cx.persistence_intervals_in_dimension(d))
        if len(bars) == 0:
            vecs.append(np.zeros(resolution[0] * resolution[1]))
        else:
            imager = PersistenceImage(bandwidth=bandwidth,
                                      resolution=list(resolution),
                                      im_range=list(im_range),
                                      weight=WEIGHTS[weight])
            vecs.append(np.asarray(imager(bars)).ravel())
    np.save(out, np.concatenate(vecs))
    return "ok"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputdir", required=True)
    p.add_argument("--outputdir", required=True)
    p.add_argument("--weight", choices=sorted(WEIGHTS), default="linear",
                   help="per-bar weight in the persistence image (gudhi's own "
                        "default is 'constant'); see the module docstring")
    p.add_argument("--resolution", type=int, nargs=2, default=[10, 10])
    p.add_argument("--im_range", type=float, nargs=4, default=[0.0, 1.25, 0.0, 1.25],
                   metavar=("BMIN", "BMAX", "PMIN", "PMAX"),
                   help="birth/persistence extent, shared by all dimensions "
                        "unless overridden below")
    p.add_argument("--bandwidth", type=float, default=1.0,
                   help="Gaussian sigma in PERSISTENCE UNITS, not pixels, "
                        "shared by all dimensions unless overridden below. The "
                        "default 1.0 (gudhi's) is ~8 px at the default range "
                        "and resolution and blurs every image to the same "
                        "shape; set it per dimension to about one pixel")
    for d in DIMS:
        p.add_argument(f"--im_range_h{d}", type=float, nargs=4, default=None,
                       metavar=("BMIN", "BMAX", "PMIN", "PMAX"),
                       help=f"override --im_range for H{d}")
        p.add_argument(f"--bandwidth_h{d}", type=float, default=None,
                       help=f"override --bandwidth for H{d}")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    # Per-dimension (bandwidth, im_range): the shared flag unless overridden.
    params = {}
    for d in DIMS:
        rng = getattr(a, f"im_range_h{d}") or a.im_range
        bw = getattr(a, f"bandwidth_h{d}")
        params[d] = (a.bandwidth if bw is None else bw, tuple(rng))

    os.makedirs(a.outputdir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.inputdir) if f.endswith(".npy"))

    print(f"  weight: {a.weight}   resolution: {a.resolution[0]}x{a.resolution[1]}")
    for d in DIMS:
        bw, rng = params[d]
        px_b = (rng[1] - rng[0]) / a.resolution[0]
        px_p = (rng[3] - rng[2]) / a.resolution[1]
        # Degeneracy is the kernel covering a large share of an AXIS, not of a
        # pixel: on a strongly anisotropic grid any bandwidth sane on the birth
        # axis is several pixels on the persistence axis, so a pixel-based test
        # would flag good configurations too.
        span = min(rng[1] - rng[0], rng[3] - rng[2])
        flag = "  <-- SMEARED" if bw > span / 3 else ""
        print(f"  H{d}: range b[{rng[0]:g},{rng[1]:g}] p[{rng[2]:g},{rng[3]:g}]  "
              f"bandwidth {bw:g} = {bw / px_b:.1f} px birth, "
              f"{bw / px_p:.1f} px persistence{flag}")

    tasks = [(f, a.inputdir, a.outputdir, tuple(a.resolution), params, a.weight)
             for f in files]

    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="PH"):
            done += f.result() == "ok"
    n_feat = 3 * a.resolution[0] * a.resolution[1]
    print(f"  {done} computed, {len(tasks) - done} cached, "
          f"{n_feat} features each, weight={a.weight} -> {a.outputdir}")


if __name__ == "__main__":
    main()
