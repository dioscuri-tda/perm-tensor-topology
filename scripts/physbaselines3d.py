r"""
Physics baselines in 3D (the "Phys" descriptor family): tensorial
Kozeny-Carman and directional critical-path radius.

Kozeny-Carman (kc_)
-------------------
Scalar form k ~ phi^3 / S^2, with phi the porosity and S the specific surface
per bulk volume. The scalar cannot carry an off-diagonal (it is invariant under
the reflection that flips the sign of k_xy), so a tensor form is added:
K_KC = k_KC * 3/2 (I - F), with F the area-weighted interface-normal tensor.
Flow is blocked along an interface normal, so for plates of normal n this
gives K ~ (I - n n), and the off-diagonal -3/2 k_KC F_ij carries a sign. The
surface is a periodic marching-cubes mesh; a voxel-face count is reported as a
cross-check (binary marching cubes reads curved surfaces ~7% high and planes
exactly).

Directional critical path (cp_)
-------------------------------
Katz-Thompson: k ~ r_c^2 / F, with r_c the largest inscribed radius at which
the pore space still percolates. For a direction w (the same nine integer
vectors as every other descriptor), r_c(w) is the largest r such that some
connected cluster of {dt >= r} contains a loop whose net displacement is
exactly w cell periods. The displacements of closed loops through a cluster on
the 3-torus form a subgroup of Z^3 (its winding lattice), so the test is
lattice membership. This is what lets r_c carry a sign: a channel along
[1,1,0] winds by (1,1,0) but not by (1,-1,0) or (1,0,0). On the three axes the
test reduces to ordinary periodic percolation. Connectivity is 6 (faces); dt
is the periodic Euclidean distance from a pore voxel to the nearest solid
voxel (`filtration_throat3d.periodic_edt`), capped at 30 voxels and asserted
not to reach the cap. This baseline is the direct non-topological counterpart
of THR, in which an H0 death time is a merging radius.

Features
--------
    kc_phi, kc_S_mc, kc_S_face      porosity; surface per bulk volume from the
                                     marching-cubes mesh and from voxel faces
    kc_k, kc_logk                    phi^3 / S_mc^2 and its log
    kc_t_xx .. kc_t_yz               k_KC * 3/2 (I - F_mc), 6 components
    cp_rc_<tag>                      directional critical radius, 9 directions
    cp_kt_<tag>                      Katz-Thompson-like r_c^2 * phi^2
                                     (Archie, m = 2, standing in for 1/F)
    cp_frac_<tag>                    pore-volume fraction of the cluster that
                                     first winds along w, at r_c(w)

Tags follow `baselines3d.tag_of`, so `a1_-1_0` is [1,-1,0] in the solver frame.

Input / output
--------------
Reads `<dataset>/structures.csv` and `<dataset>/structures/<filename>.raw`;
writes one CSV with a `sample_id` column and the features above.

Usage
-----
    python scripts/physbaselines3d.py --self-test
    python scripts/physbaselines3d.py --dataset DATA/aniso3d \
        --output DESC/aniso3d/phys.csv [--workers 24] [--limit N]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io3d import load_structure                   # noqa: E402
from baselines3d import DIRECTIONS, tag_of                  # noqa: E402
from filtration_throat3d import periodic_edt                # noqa: E402

CONN6 = np.zeros((3, 3, 3), dtype=bool)
CONN6[1, 1, :] = CONN6[1, :, 1] = CONN6[:, 1, 1] = True
DT_CAP = 30.0


# ---------------------------------------------------------------------------
# Kozeny-Carman
# ---------------------------------------------------------------------------

def surface_mesh(solid):
    """
    Periodic marching-cubes mesh of the solid/pore interface.

    One wrapped layer on the high side gives N+1 points per axis, i.e. exactly
    N cells, so every interface on the torus is meshed once.  Returns the total
    area and the area-weighted normal tensor F_ij = sum a n_i n_j / sum a in
    the SOLVER frame (x, y, z).
    """
    g = np.pad(solid.astype(np.float32), ((0, 1),) * 3, mode="wrap")
    verts, faces, _, _ = marching_cubes(g, level=0.5)          # verts in (z, y, x)
    tri = verts[faces]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(cr, axis=1)                             # 2 * area
    keep = a2 > 0
    n = cr[keep] / a2[keep, None]
    w = a2[keep]
    n = n[:, ::-1]                                              # -> (x, y, z)
    F = (n.T * w) @ n / w.sum()
    return 0.5 * a2.sum(), F


def face_area(solid):
    """Number of solid/pore voxel faces on the torus (unit faces)."""
    s = solid.astype(np.int8)
    return int(sum((s != np.roll(s, 1, axis=ax)).sum() for ax in range(3)))


def kozeny_carman(solid):
    N3 = solid.size
    phi = 1.0 - solid.mean()
    area, F = surface_mesh(solid)
    S = area / N3
    kc = phi ** 3 / S ** 2
    T = kc * 1.5 * (np.eye(3) - F)
    row = {"kc_phi": phi, "kc_S_mc": S, "kc_S_face": face_area(solid) / N3,
           "kc_k": kc, "kc_logk": np.log(kc)}
    for (i, j), nm in zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
                          ["xx", "yy", "zz", "xy", "xz", "yz"]):
        row[f"kc_t_{nm}"] = T[i, j]
    return row


# ---------------------------------------------------------------------------
# Winding lattices on the torus
# ---------------------------------------------------------------------------

def lattice_basis(vecs):
    """Echelon (Hermite-like) basis of the Z-span of integer 3-vectors."""
    M = [list(v) for v in vecs if any(v)]
    basis = []
    for c in range(3):
        rows = [r for r in M if r[c] != 0]
        rest = [r for r in M if r[c] == 0]
        while len(rows) > 1:
            rows.sort(key=lambda r: abs(r[c]))
            p = rows[0]
            new = [p]
            for r in rows[1:]:
                q = r[c] // p[c]
                r2 = [r[i] - q * p[i] for i in range(3)]
                if r2[c] != 0:
                    new.append(r2)
                elif any(r2):
                    rest.append(r2)
            rows = new
        if rows:
            basis.append((c, rows[0]))
        M = rest
    return basis


def in_lattice(basis, w):
    w = list(w)
    for c, row in basis:
        if w[c] % row[c]:
            return False
        q = w[c] // row[c]
        w = [w[i] - q * row[i] for i in range(3)]
    return not any(w)


def winding_clusters(mask):
    """
    Label `mask` on the 3-torus.  Returns (lab, root_of_label, lattices) where
    lattices maps a root label to the echelon basis of its winding lattice in
    the solver frame (x, y, z).

    scipy labels the open cube; the periodic faces are then glued by a union-
    find whose potentials are cell offsets, so a face edge that closes a cycle
    contributes that cycle's net displacement.
    """
    lab, n = label(mask, structure=CONN6)
    if n == 0:
        return lab, np.zeros(1, dtype=np.int64), {}
    edges = []
    for ax, shift in ((2, (1, 0, 0)), (1, (0, 1, 0)), (0, (0, 0, 1))):
        a = np.take(lab, -1, axis=ax).ravel()
        b = np.take(lab, 0, axis=ax).ravel()
        m = (a > 0) & (b > 0)
        if m.any():
            ab = np.unique(np.stack([a[m], b[m]], 1), axis=0)
            edges.extend((int(x), int(y), shift) for x, y in ab)

    parent = list(range(n + 1))
    pot = [(0, 0, 0)] * (n + 1)           # cell offset of node relative to parent
    cycles = {}

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root = x
        acc = (0, 0, 0)                    # compress, accumulating from the top
        for node in reversed(path):
            p = pot[node]
            acc = (acc[0] + p[0], acc[1] + p[1], acc[2] + p[2])
            pot[node] = acc
            parent[node] = root
        return root

    def potential(x):
        find(x)
        return pot[x] if parent[x] != x else (0, 0, 0)

    for a, b, s in edges:
        ra, rb = find(a), find(b)
        pa, pb = potential(a), potential(b)
        # copy of b adjacent to a sits at cell(a) + s
        d = (pa[0] + s[0] - pb[0], pa[1] + s[1] - pb[1], pa[2] + s[2] - pb[2])
        if ra == rb:
            if any(d):
                cycles.setdefault(ra, set()).add(d)
        else:
            parent[rb] = ra
            pot[rb] = d
            if rb in cycles:
                cycles.setdefault(ra, set()).update(cycles.pop(rb))

    root = np.array([find(i) for i in range(n + 1)], dtype=np.int64)
    lattices = {r: lattice_basis(v) for r, v in cycles.items()}
    return lab, root, lattices


def critical_path(solid, dt=None):
    """r_c(w) and the winding cluster's pore fraction, for the nine directions."""
    pore = solid == 0
    if dt is None:
        dt = periodic_edt(solid, fg=0, rcap=DT_CAP)
    assert dt.max() < DT_CAP, f"dt reached the cap {DT_CAP}"
    n_pore = pore.sum()
    levels = np.unique(dt[pore])[::-1]                  # descending radii
    rc = {w: 0.0 for w in DIRECTIONS}
    frac = {w: 0.0 for w in DIRECTIONS}
    todo = set(DIRECTIONS)
    for r in levels:
        lab, root, lat = winding_clusters(dt >= r)
        hit = [(w, rt) for w in todo for rt, b in lat.items() if in_lattice(b, w)]
        if hit:
            croot = root[lab]
            for w, rt in hit:
                if w in todo:
                    rc[w] = float(r)
                    frac[w] = float((croot == rt).sum()) / n_pore   # rt >= 1: never the background
                    todo.discard(w)
        if not todo:
            break
    phi = pore.mean()
    row = {}
    for w in DIRECTIONS:
        t = tag_of(w)
        row[f"cp_rc_{t}"] = rc[w]
        row[f"cp_kt_{t}"] = rc[w] ** 2 * phi ** 2
        row[f"cp_frac_{t}"] = frac[w]
    return row


def _one(args):
    sample_id, path = args
    s = load_structure(path)
    row = {"sample_id": sample_id}
    row.update(kozeny_carman(s))
    row.update(critical_path(s))
    return row


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    N = 40
    Z, Y, X = np.mgrid[0:N, 0:N, 0:N]

    def tube(axis_fn, R):
        """solid everywhere except a tube of radius R; axis_fn -> squared dist"""
        return (axis_fn() > R * R).astype(np.uint8)

    # straight tube along x, radius 5, centred at y = z = 20
    sx = tube(lambda: (Y - 20) ** 2 + (Z - 20) ** 2, 5)
    cx = critical_path(sx)
    check("x-tube: r_c(x) > 0, r_c(y) = r_c(z) = 0",
          cx["cp_rc_a1_0_0"] > 3 and cx["cp_rc_a0_1_0"] == 0 and cx["cp_rc_a0_0_1"] == 0,
          f"rc x/y/z = {cx['cp_rc_a1_0_0']:.2f}/{cx['cp_rc_a0_1_0']:.2f}/{cx['cp_rc_a0_0_1']:.2f}")
    check("x-tube: no diagonal winding",
          all(cx[f"cp_rc_{tag_of(w)}"] == 0 for w in DIRECTIONS[3:]))
    dtx = periodic_edt(sx, fg=0, rcap=DT_CAP)
    core = (dtx >= cx["cp_rc_a1_0_0"]).sum() / (sx == 0).sum()
    check("x-tube: winding cluster at r_c is the tube core {dt >= r_c}",
          abs(cx["cp_frac_a1_0_0"] - core) < 1e-12, f"{cx['cp_frac_a1_0_0']:.4f} vs {core:.4f}")

    # tube along [1,1,0]: distance to the line x - y = 0 (mod N) in the xy plane,
    # taken periodically, any z within R of 20
    def diag(sign):
        u = (X - sign * Y) % N
        u = np.minimum(u, N - u) / np.sqrt(2.0)
        return u ** 2 + (Z - 20) ** 2
    sp = tube(lambda: diag(+1), 4)
    sm = tube(lambda: diag(-1), 4)
    cp, cm = critical_path(sp), critical_path(sm)
    check("[1,1,0] tube winds along (1,1,0) only",
          cp["cp_rc_a1_1_0"] > 2 and cp["cp_rc_a1_-1_0"] == 0 and cp["cp_rc_a1_0_0"] == 0,
          f"rc(1,1,0)={cp['cp_rc_a1_1_0']:.2f} rc(1,-1,0)={cp['cp_rc_a1_-1_0']:.2f} "
          f"rc(1,0,0)={cp['cp_rc_a1_0_0']:.2f}")
    check("[1,-1,0] tube is its mirror: the sign flips",
          cm["cp_rc_a1_-1_0"] == cp["cp_rc_a1_1_0"] and cm["cp_rc_a1_1_0"] == 0)

    # two tubes, x (wide) and y (narrow), crossing: lattice is generated by both,
    # so both diagonals appear at the narrow radius
    sxy = (sx.astype(bool) & tube(lambda: (X - 10) ** 2 + (Z - 20) ** 2, 3).astype(bool)).astype(np.uint8)
    cxy = critical_path(sxy)
    check("crossed tubes: r_c(x) > r_c(y) > 0 and both diagonals at r_c(y)",
          cxy["cp_rc_a1_0_0"] > cxy["cp_rc_a0_1_0"] > 0
          and cxy["cp_rc_a1_1_0"] == cxy["cp_rc_a1_-1_0"] == cxy["cp_rc_a0_1_0"],
          f"x={cxy['cp_rc_a1_0_0']:.2f} y={cxy['cp_rc_a0_1_0']:.2f} "
          f"d+={cxy['cp_rc_a1_1_0']:.2f} d-={cxy['cp_rc_a1_-1_0']:.2f}")

    # axis permutation x <-> z moves r_c(x) to r_c(z)
    sz = np.ascontiguousarray(np.transpose(sx, (2, 1, 0)))
    cz = critical_path(sz)
    check("swapping solver x and z moves r_c(x) to r_c(z)",
          cz["cp_rc_a0_0_1"] == cx["cp_rc_a1_0_0"] and cz["cp_rc_a1_0_0"] == 0)

    # lattice arithmetic
    b = lattice_basis([(2, 0, 0), (0, 1, 0)])
    check("lattice: (2,0,0),(0,1,0) contains (2,3,0), not (1,0,0)",
          in_lattice(b, (2, 3, 0)) and not in_lattice(b, (1, 0, 0)))
    b = lattice_basis([(1, 1, 0), (1, -1, 0)])
    check("lattice: (1,1,0),(1,-1,0) contains (2,0,0) and (0,2,0), not (1,0,0)",
          in_lattice(b, (2, 0, 0)) and in_lattice(b, (0, 2, 0)) and not in_lattice(b, (1, 0, 0)))

    # Kozeny-Carman: x-normal plates. Mesh normals along x; tensor blocks x.
    plates = np.zeros((N, N, N), dtype=np.uint8)
    for x0 in range(0, N, 10):
        plates[:, :, x0:x0 + 4] = 1
    k = kozeny_carman(plates)
    check("plates: S per volume = 2 faces per period / 10 voxels",
          abs(k["kc_S_mc"] - 0.2) < 1e-9 and abs(k["kc_S_face"] - 0.2) < 1e-9,
          f"S_mc={k['kc_S_mc']:.4f} S_face={k['kc_S_face']:.4f}")
    check("plates: KC tensor kills xx, keeps yy = zz, no off-diagonal",
          abs(k["kc_t_xx"]) < 1e-9 and abs(k["kc_t_yy"] - k["kc_t_zz"]) < 1e-9 and k["kc_t_yy"] > 0
          and max(abs(k[f"kc_t_{c}"]) for c in ("xy", "xz", "yz")) < 1e-9,
          f"xx={k['kc_t_xx']:.2e} yy={k['kc_t_yy']:.4f}")

    # a sphere: mesh area ~ 4 pi R^2, isotropic normal tensor
    R = 12
    ball = ((X - 20) ** 2 + (Y - 20) ** 2 + (Z - 20) ** 2 <= R * R).astype(np.uint8)
    area, F = surface_mesh(ball)
    # binary marching cubes overestimates CURVED surfaces (+6.9% here; digital
    # planes along [1,0,0], [1,1,0], [1,1,1] are exact). Gaussian pre-smoothing
    # would remove the bias but erodes 2-voxel throats, so the mesh is left
    # unsmoothed and parameter-free.
    check("sphere: marching-cubes area within 8% of 4 pi R^2",
          abs(area / (4 * np.pi * R * R) - 1) < 0.08, f"ratio {area / (4 * np.pi * R * R):.4f}")
    check("sphere: normal tensor ~ I/3",
          np.abs(F - np.eye(3) / 3).max() < 0.01, f"max dev {np.abs(F - np.eye(3) / 3).max():.4f}")

    # a tilted structure must give the off-diagonal the right sign: plates with
    # normal along [1,1,0] block flow along it, so K_xy < 0
    tilt = (((X + Y) % 8) < 3).astype(np.uint8)     # 8 divides N: periodic
    kt = kozeny_carman(tilt)
    tilt_m = (((X - Y) % 8) < 3).astype(np.uint8)
    ktm = kozeny_carman(tilt_m)
    check("[1,1,0]-normal plates: kc_t_xy < 0, mirror flips it",
          kt["kc_t_xy"] < 0 < ktm["kc_t_xy"] and abs(kt["kc_t_xy"] + ktm["kc_t_xy"]) < 1e-9,
          f"{kt['kc_t_xy']:.4f} / {ktm['kc_t_xy']:.4f}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--output")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.output):
        p.error("--dataset and --output are required (or use --self-test)")

    df = pd.read_csv(os.path.join(a.dataset, "structures.csv"))
    if a.limit:
        df = df.head(a.limit)
    tasks = [(r.sample_id, os.path.join(a.dataset, "structures", r.filename))
             for r in df.itertuples()]

    from tqdm import tqdm
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="physbaselines",
                      mininterval=30):
            rows.append(f.result())

    out = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"  {len(out)} rows x {out.shape[1] - 1} features -> {a.output}")
    print(f"  NaNs: {int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
