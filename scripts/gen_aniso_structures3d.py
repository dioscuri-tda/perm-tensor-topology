r"""
Generate 3D periodic porous structures with controlled flow anisotropy.

This produces the synthetic anisotropic dataset (`DATA/aniso3d`: 5000
structures, 80^3 voxels) used to train and test permeability tensor models.

Model
-----
A random trigonometric field is thresholded to a target porosity:

    S(r) = sqrt(2/N) Σ_i cos(q_i · r + φ_i),   solid where S > t

Wavevectors are drawn isotropically in a band around k0, squashed by a
volume-preserving ellipsoidal map, and rounded to integers:

    q = R diag(s1, s2, s3) Rᵀ q_iso ,   s = (AB)^(1/3) · (1/A, 1/B, 1)

Integer wavevectors make the unit cell exactly periodic, which the permeability
tensor of a periodic cell requires. R is a uniformly random rotation of SO(3)
(Shoemake's quaternion method), so the principal frame is off-grid and all six
tensor components are non-zero. Components along the first principal axis e1
shrink the most, so solid features elongate along e1:

    A  = elongation ratio e1 : e3
    B  = elongation ratio e2 : e3,   drawn in [1, A]

    B = 1   rod   (prolate)   k1 > k2 = k3
    B = A   plate (oblate)    k1 = k2 > k3
    else    triaxial          three distinct principal permeabilities

`shape = (B-1)/(A-1)` is recorded as a 0 (rod) .. 1 (plate) coordinate. Samples
cycle through three strata: iso (A = 1), mod (A in [1.5, 2.5]) and strong
(A in [2.5, 4]). Only structures whose pore space percolates (6-connected, on the periodic torus) along x, y and z are kept.
Each sample uses its own RNG seeded with `seed + sample_id`, so any sample can
be regenerated alone and results do not depend on the installed scipy.

Frame convention
----------------
Arrays are indexed ``[z, y, x]`` with x varying fastest, matching
`structure_io3d` and the C++ solver, so e1 is directly comparable with the
principal eigenvector reported by the solver. `--self-test` checks this.

Outputs
-------
    <output>/structures/sample_NNNNNN_str=..._A=..._B=..._k0=..._por=....raw
    <output>/structures.csv   one row per kept sample: generator parameters
                              (A, B, shape, e1, e2, k0, n_modes, porosity, ...)
    <output>/filelist.txt     paths of the .raw files
    <output>/gen.log

The generator parameters are for stratification and analysis only; they are
not model features. Join on `sample_id`; do not parse filenames.

Usage
-----
    python scripts/gen_aniso_structures3d.py --self-test
    python scripts/gen_aniso_structures3d.py --output DATA/aniso3d --n_samples 5000
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm

from structure_io3d import save_structure

# Strata: (name, min primary axis ratio, max primary axis ratio).
STRATA = [
    ("iso",    1.0, 1.0),
    ("mod",    1.5, 2.5),
    ("strong", 2.5, 4.0),
]

# 6-connectivity: a face-diagonal pinch has zero cross-section and conducts
# essentially nothing under D3Q19 bounce-back, so counting it would admit
# structures whose permeability is numerically zero and which then burn the
# solver's whole step budget without converging.
_CONN6 = generate_binary_structure(3, 1)

_PTS = None


def _init_worker(size):
    global _PTS
    _PTS = np.linspace(0.0, 1.0, size, endpoint=False)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def random_rotation(rng):
    """
    A uniformly distributed rotation of SO(3) (Shoemake 1992).

    Drawn from `rng` rather than `scipy.spatial.transform.Rotation.random` so a
    sample's content depends only on `seed + sample_id` and not on the scipy
    version.
    """
    u1, u2, u3 = rng.random(3)
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    q = np.array([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2),
                  b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3)])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def squash_matrix(aniso_a, aniso_b, R):
    """Volume-preserving ellipsoidal map; columns of R are the principal axes."""
    scale = (aniso_a * aniso_b) ** (1.0 / 3.0)
    s = scale * np.array([1.0 / aniso_a, 1.0 / aniso_b, 1.0])
    return R @ np.diag(s) @ R.T


def sample_wavevectors(rng, n_modes, k0, dk, aniso_a, aniso_b, R):
    """Integer wavevectors (qx, qy, qz) from an ellipsoidal band."""
    # isotropic directions on the sphere
    v = rng.normal(size=(n_modes, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    mag = rng.uniform(max(0.5, k0 - dk), k0 + dk, n_modes)
    q = v * mag[:, None]

    if not (aniso_a == 1.0 and aniso_b == 1.0):
        q = q @ squash_matrix(aniso_a, aniso_b, R).T

    # Integer components are what make the cell exactly periodic.
    q = np.rint(q).astype(int)
    return q[np.any(q != 0, axis=1)]


def build_field(pts, q_int, rng):
    """
    sqrt(2/N) Σ cos(q·r + φ), accumulated one mode at a time.

    q·r is separable over the axes, so the phase is an outer sum of three 1D
    terms and no (size, size, size, N) array is ever materialised (that would
    be 1.3 GB per worker at size 128).
    """
    size = len(pts)
    n = len(q_int)
    twopi = 2.0 * np.pi
    phis = rng.uniform(0.0, np.pi, n)

    acc = np.zeros((size, size, size))
    phase = np.empty_like(acc)
    for (qx, qy, qz), phi in zip(q_int, phis):
        # arrays are [z, y, x]
        np.add.outer(np.add.outer(twopi * qz * pts, twopi * qy * pts),
                     twopi * qx * pts, out=phase)
        phase += phi
        acc += np.cos(phase, out=phase)
    return np.sqrt(2.0 / n) * acc


def threshold_for_porosity(field, porosity):
    """Exact quantile threshold: solid where field > t, pore fraction = porosity."""
    return float(np.quantile(field, porosity))


def percolates(solid):
    """
    (perc_x, perc_y, perc_z) for the pore phase on the periodic torus.

    Tests whether a pore cluster spans a 3x3x3 tiling face to face, which a
    wrapping cluster always does and a merely large local one cannot.
    """
    pore = np.tile(~solid, (3, 3, 3))
    lab, n = label(pore, structure=_CONN6)
    if n == 0:
        return False, False, False
    spans = []
    for axis in (2, 1, 0):                   # x, y, z
        lo = set(np.take(lab, 0, axis=axis).ravel())
        hi = set(np.take(lab, -1, axis=axis).ravel())
        spans.append(bool((lo & hi) - {0}))
    return tuple(spans)


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

def gen_one(sample_id, cfg):
    rng = np.random.default_rng(cfg["seed"] + sample_id)
    name, a_lo, a_hi = STRATA[sample_id % len(STRATA)]

    for attempt in range(cfg["max_attempts"]):
        aniso_a = float(rng.uniform(a_lo, a_hi))
        # B in [1, A]: 1 = rod, A = plate.  Drawn even when A == 1 so that the
        # iso stratum consumes the same number of random draws as the others and
        # a sample's content stays a function of its index alone.
        aniso_b = float(rng.uniform(1.0, aniso_a)) if aniso_a > 1.0 else 1.0
        R = random_rotation(rng)
        k0 = float(rng.uniform(cfg["k0_min"], cfg["k0_max"]))
        n_modes = int(rng.integers(cfg["modes_min"], cfg["modes_max"] + 1))
        porosity = float(rng.uniform(cfg["por_min"], cfg["por_max"]))

        q = sample_wavevectors(rng, n_modes, k0, cfg["dk"], aniso_a, aniso_b, R)
        if len(q) < 5:
            continue

        field = build_field(_PTS, q, rng)
        thresh = threshold_for_porosity(field, porosity)
        solid = field > thresh

        px, py, pz = percolates(solid)
        if not (px and py and pz):
            continue

        por = float(1.0 - solid.mean())
        shape = (aniso_b - 1.0) / (aniso_a - 1.0) if aniso_a > 1.0 else 0.0
        e1 = R[:, 0]
        fname = (f"sample_{sample_id:06d}_str={name}_A={aniso_a:.2f}"
                 f"_B={aniso_b:.2f}_k0={k0:.2f}_por={por:.4f}.raw")

        save_structure(os.path.join(cfg["structures_dir"], fname), solid)

        return {
            "sample_id": sample_id,
            "filename": fname,
            "stratum": name,
            "aniso_A": aniso_a,
            "aniso_B": aniso_b,
            "shape": shape,          # 0 = rod, 1 = plate
            "e1_x": e1[0], "e1_y": e1[1], "e1_z": e1[2],
            "e2_x": R[0, 1], "e2_y": R[1, 1], "e2_z": R[2, 1],
            "k0": k0,
            "n_modes": len(q),
            "porosity": por,
            "porosity_target": porosity,
            "threshold": thresh,
            "attempts": attempt + 1,
        }

    logging.warning("sample %d (%s): no percolating structure in %d attempts",
                    sample_id, name, cfg["max_attempts"])
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test(size):
    """Frame, anisotropy and periodicity checks that need no solver."""
    size = min(size, 48)                     # keep the self-test quick
    pts = np.linspace(0.0, 1.0, size, endpoint=False)
    rng = np.random.default_rng(0)
    ok = True

    def check(label_, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label_}{(' ' + detail) if detail else ''}")

    # --- frame: a pure mode must vary along the axis it names -----------------
    for axis_name, qvec, arr_axis in (("x", [4, 0, 0], 2),
                                      ("y", [0, 4, 0], 1),
                                      ("z", [0, 0, 4], 0)):
        f = build_field(pts, np.array([qvec]), np.random.default_rng(1))
        v = [f.var(axis=a).mean() for a in (0, 1, 2)]
        # variance *along* an axis survives when we average over the other two
        along = np.var(f.mean(axis=tuple(a for a in (0, 1, 2) if a != arr_axis)))
        others = [np.var(f.mean(axis=tuple(a for a in (0, 1, 2) if a != b)))
                  for b in (0, 1, 2) if b != arr_axis]
        check(f"q along {axis_name} varies over array axis {arr_axis}",
              along > 100 * max(others),
              f"var {along:.4f} vs {max(others):.2e}")

    # --- the squash is volume preserving and R is a proper rotation ----------
    R = random_rotation(rng)
    check("R is orthonormal", np.allclose(R @ R.T, np.eye(3), atol=1e-12))
    check("det R = +1 (proper rotation, not a reflection)",
          abs(np.linalg.det(R) - 1.0) < 1e-12)
    M = squash_matrix(3.0, 1.7, R)
    check("det(squash) = 1 (volume preserving)",
          abs(np.linalg.det(M) - 1.0) < 1e-12, f"det {np.linalg.det(M):.12f}")

    # --- rod vs plate --------------------------------------------------------
    s_rod = np.linalg.eigvalsh(squash_matrix(3.0, 1.0, np.eye(3)))
    s_plate = np.linalg.eigvalsh(squash_matrix(3.0, 3.0, np.eye(3)))
    check("B=1 is a rod (two equal large wavenumbers)",
          abs(s_rod[1] - s_rod[2]) < 1e-12 and s_rod[0] < s_rod[1])
    check("B=A is a plate (two equal small wavenumbers)",
          abs(s_plate[0] - s_plate[1]) < 1e-12 and s_plate[1] < s_plate[2])

    # --- elongation really follows e1 ---------------------------------------
    for axis_name, col in (("x", 0), ("y", 1), ("z", 2)):
        Rax = np.eye(3)[:, [col] + [c for c in range(3) if c != col]]
        if np.linalg.det(Rax) < 0:
            Rax[:, [1, 2]] = Rax[:, [2, 1]]
        q = sample_wavevectors(rng, 20000, 8.0, 1.0, 3.5, 1.0, Rax)
        rms = np.sqrt((q ** 2).mean(axis=0))     # (qx, qy, qz)
        smallest = int(np.argmin(rms))
        check(f"e1 = +{axis_name} shrinks the {axis_name} wavenumbers most",
              smallest == col, f"rms|q| = {np.round(rms, 2)}")

    # --- periodicity: the field is continuous across every wrap --------------
    R = random_rotation(np.random.default_rng(7))
    q = sample_wavevectors(np.random.default_rng(3), 40, 6.0, 1.0, 2.0, 1.4, R)
    f = build_field(pts, q, np.random.default_rng(2))
    wrap = max(np.abs(np.take(f, 0, a) - np.take(f, -1, a)).max() for a in (0, 1, 2))
    step = max(np.abs(np.diff(f, axis=a)).max() for a in (0, 1, 2))
    check("periodic wrap jump <= 1.5x largest interior step",
          wrap <= 1.5 * step, f"{wrap:.4f} vs {step:.4f}")

    # --- percolation ---------------------------------------------------------
    solid = np.zeros((size, size, size), dtype=bool)
    solid[:, :, size // 2] = True            # slab normal to x, blocks x only
    px, py, pz = percolates(solid)
    check("solid slab normal to x blocks x only",
          (not px) and py and pz, f"perc = ({px}, {py}, {pz})")

    solid = np.zeros((size, size, size), dtype=bool)
    solid[size // 2, :, :] = True            # slab normal to z
    px, py, pz = percolates(solid)
    check("solid slab normal to z blocks z only",
          px and py and (not pz), f"perc = ({px}, {py}, {pz})")

    # --- determinism ---------------------------------------------------------
    cfg = dict(structures_dir=None, seed=0, k0_min=5.0, k0_max=7.0, dk=1.0,
               modes_min=20, modes_max=30, por_min=0.7, por_max=0.85,
               max_attempts=20)

    def field_for(sample_id):
        r = np.random.default_rng(cfg["seed"] + sample_id)
        a = 3.0
        b = float(r.uniform(1.0, a))
        Rr = random_rotation(r)
        k0 = float(r.uniform(cfg["k0_min"], cfg["k0_max"]))
        nm = int(r.integers(cfg["modes_min"], cfg["modes_max"] + 1))
        r.uniform(cfg["por_min"], cfg["por_max"])
        qq = sample_wavevectors(r, nm, k0, cfg["dk"], a, b, Rr)
        return build_field(pts, qq, r)

    check("same seed reproduces the field bit for bit",
          np.array_equal(field_for(5), field_for(5)))
    check("different sample_id gives a different field",
          not np.array_equal(field_for(5), field_for(6)))

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--size", type=int, default=80,
                   help="cubic domain edge in voxels; LBM cost grows roughly as size^5")
    p.add_argument("--seed", type=int, default=0,
                   help="added to the sample index to seed each structure")
    p.add_argument("--k0_min", type=float, default=3.0)
    p.add_argument("--k0_max", type=float, default=7.0)
    p.add_argument("--dk", type=float, default=1.0)
    p.add_argument("--modes_min", type=int, default=20)
    p.add_argument("--modes_max", type=int, default=80)
    p.add_argument("--porosity_min", type=float, default=0.65)
    p.add_argument("--porosity_max", type=float, default=0.90)
    p.add_argument("--max_attempts", type=int, default=200)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--self-test", action="store_true", dest="selftest")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(self_test(args.size))
    if not args.output:
        p.error("--output is required (or use --self-test)")

    structures_dir = os.path.join(args.output, "structures")
    os.makedirs(structures_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(args.output, "gen.log")),
                  logging.StreamHandler()])
    logging.info("size=%d n_samples=%d seed=%d", args.size, args.n_samples, args.seed)

    cfg = {
        "structures_dir": structures_dir,
        "seed": args.seed,
        "k0_min": args.k0_min, "k0_max": args.k0_max, "dk": args.dk,
        "modes_min": args.modes_min, "modes_max": args.modes_max,
        "por_min": args.porosity_min, "por_max": args.porosity_max,
        "max_attempts": args.max_attempts,
    }

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.size,)) as pool:
        futs = [pool.submit(gen_one, i, cfg) for i in range(args.n_samples)]
        for f in tqdm(as_completed(futs), total=len(futs), desc="generating"):
            r = f.result()
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    df.to_csv(os.path.join(args.output, "structures.csv"), index=False)
    with open(os.path.join(args.output, "filelist.txt"), "w") as fh:
        for n in df.filename:
            fh.write(os.path.join(structures_dir, n) + "\n")

    logging.info("kept %d/%d structures", len(df), args.n_samples)
    print(df.groupby("stratum")[["porosity", "aniso_A", "aniso_B", "shape",
                                 "k0", "attempts"]].describe().T.to_string())


if __name__ == "__main__":
    main()
