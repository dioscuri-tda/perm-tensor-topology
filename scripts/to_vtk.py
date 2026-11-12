"""
Convert velocity or scalar fields stored as .npy into a legacy VTK file that
ParaView opens directly.

Several named arrays on one grid can go into a single file (`write_vtk_multi`,
or `--field NAME=FILE`, repeatable); this is how `solve_velocity3d.py` writes
the three per-direction velocity fields together with the `solid` mask. Vector
fields also get a `<name>_magnitude` scalar unless --no-magnitude is given.
`--tile N` replicates the periodic cell so that streamlines run on across the
seams.

Array convention
----------------
Axes run slowest to fastest, ``(y, x)`` in 2D and ``(z, y, x)`` in 3D, with an
optional trailing component axis:

    (ny, nx)          scalar on a 2D grid
    (ny, nx, 2 or 3)  vector on a 2D grid
    (nz, ny, nx)      scalar on a 3D grid
    (nz, ny, nx, 3)   vector on a 3D grid

This is the orientation `structure_io3d.load_structure` returns, so a field and
its structure mask line up without transposing either. VTK STRUCTURED_POINTS
stores points with x varying fastest, then y, then z, which is exactly C order
for an array indexed ``(z, y, x)``, so `reshape(-1)` is already the right
sequence. Getting this wrong would transpose the field, which for an
anisotropic structure still looks plausible; the self-test checks it.

Output is binary by default (legacy VTK binary is big-endian regardless of
host). `--ascii` is readable but about 15x larger.

Usage
-----
    python scripts/to_vtk.py field.npy field.vtk
    python scripts/to_vtk.py field.npy field.vtk --solid solid.npy
    python scripts/to_vtk.py -o all.vtk --solid solid.npy \\
           --field velocity_fx=s0.fx.npy --field velocity_fy=s0.fy.npy
    python scripts/to_vtk.py --self-test
"""

import argparse
import sys

import numpy as np

__all__ = ["write_vtk", "write_vtk_multi", "tile_field"]


def _grid_shape(arr, kind):
    """Split an array shape into (grid dims, n components)."""
    if kind == "vector":
        if arr.ndim not in (3, 4) or arr.shape[-1] not in (2, 3):
            raise ValueError(
                f"vector field must be (ny,nx,2|3) or (nz,ny,nx,3), got {arr.shape}")
        return arr.shape[:-1], arr.shape[-1]
    if arr.ndim not in (2, 3):
        raise ValueError(f"scalar field must be (ny,nx) or (nz,ny,nx), got {arr.shape}")
    return arr.shape, 1


def _guess_kind(arr):
    """Vector iff there is a trailing axis of length 2 or 3 above a 2D grid."""
    if arr.ndim in (3, 4) and arr.shape[-1] in (2, 3):
        return "vector"
    return "scalar"


def _dims(grid):
    """VTK DIMENSIONS is (nx, ny, nz); our grid is (ny, nx) or (nz, ny, nx)."""
    if len(grid) == 2:
        ny, nx = grid
        return nx, ny, 1
    nz, ny, nx = grid
    return nx, ny, nz


def _as_vtk_vectors(arr, ncomp):
    """Flatten to (npoints, 3), padding a 2-component field with vz = 0."""
    flat = arr.reshape(-1, ncomp)
    if ncomp == 3:
        return flat
    out = np.zeros((flat.shape[0], 3), dtype=flat.dtype)
    out[:, :2] = flat
    return out


def tile_field(arr, n, kind):
    """
    Replicate a field n times along each grid axis.

    The LBM cell is periodic, but a VTK dataset is not: `vtkStreamTracer` stops
    at the bounds, so a streamline leaving the top edge does not re-enter at the
    bottom. The tiled copies carry the true neighbour relation, so the lines run
    on continuously across the seams.

    No off-by-one: the period is exactly the node count (node nx-1's neighbour is
    node 0), so plain replication puts the right value next to the right value at
    every seam.  Only the outer boundary of the whole tiled block is artificial,
    and that is unavoidable.
    """
    if n == 1:
        return arr
    if n < 1:
        raise ValueError(f"--tile must be >= 1, got {n}")
    ndim = arr.ndim - (1 if kind == "vector" else 0)
    reps = (n,) * ndim + ((1,) if kind == "vector" else ())
    return np.tile(arr, reps)


def _field_arrays(name, field, kind, magnitude):
    """The (declaration, payload) pairs one named field contributes."""
    if any(c.isspace() for c in name) or not name:
        raise ValueError(f"array name {name!r} must be non-empty and whitespace-free "
                         f"(legal VTK names are whitespace-delimited)")
    _, ncomp = _grid_shape(field, kind)
    if kind == "vector":
        out = [([f"VECTORS {name} float"],
                _as_vtk_vectors(field, ncomp).astype(np.float32))]
        if magnitude:
            mag = np.linalg.norm(field.reshape(-1, ncomp), axis=1)
            out.append(([f"SCALARS {name}_magnitude float 1", "LOOKUP_TABLE default"],
                        mag.astype(np.float32)))
        return out
    return [([f"SCALARS {name} float 1", "LOOKUP_TABLE default"],
             field.reshape(-1).astype(np.float32))]


def write_vtk_multi(path, fields, solid=None, magnitude=True, ascii_mode=False,
                    kinds=None, tile=1, title="perm-tensor-topology"):
    """
    Write several named arrays on one grid to a single legacy VTK file.

    fields     mapping {name: array}; every array must share the same grid, and
               insertion order is the order they appear in the file (the first
               vector becomes VTK's active one, which is what ParaView preselects).
    solid      optional mask on the same grid, added as a scalar array so the
               geometry can be shown alongside the flow (a Threshold in ParaView).
    magnitude  for each vector field, also write |v| as `<name>_magnitude`;
               saves reaching for a Calculator filter just to colour by speed.
    kinds      optional {name: "vector"|"scalar"} to override the per-field guess.
    tile       replicate the periodic cell n times along each axis, so streamlines
               cross the seams instead of stopping at them (`tile_field`).  Costs
               n^2 in 2D and n^3 in 3D, in both file size and ParaView memory.
    """
    if not fields:
        raise ValueError("no fields given")
    kinds = kinds or {}

    grid = None
    arrays = []          # (declaration lines, flat float32/uint8 payload)
    for name, field in fields.items():
        field = np.asarray(field)
        kind = kinds.get(name) or _guess_kind(field)
        g, _ = _grid_shape(field, kind)
        if grid is None:
            grid = g
        elif tuple(g) != tuple(grid):
            raise ValueError(f"field {name!r} has grid {g}, expected {grid}")
        arrays.extend(_field_arrays(name, tile_field(field, tile, kind),
                                    kind, magnitude))
    grid = tuple(d * tile for d in grid)

    nx, ny, nz = _dims(grid)
    npoints = nx * ny * nz

    if solid is not None:
        solid = np.asarray(solid)
        if tuple(d * tile for d in solid.shape) != tuple(grid):
            raise ValueError(f"solid shape {solid.shape} does not match grid "
                             f"{tuple(d // tile for d in grid)}")
        arrays.append((["SCALARS solid unsigned_char 1", "LOOKUP_TABLE default"],
                       tile_field(solid, tile, "scalar").reshape(-1).astype(np.uint8)))

    header = (
        "# vtk DataFile Version 3.0\n"
        f"{title}\n"
        f"{'ASCII' if ascii_mode else 'BINARY'}\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx} {ny} {nz}\n"
        "ORIGIN 0 0 0\n"
        "SPACING 1 1 1\n"
        f"POINT_DATA {npoints}\n"
    )

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for decl, payload in arrays:
            fh.write(("\n".join(decl) + "\n").encode("ascii"))
            if ascii_mode:
                rows = payload.reshape(payload.shape[0], -1)
                fmt = "%d" if payload.dtype == np.uint8 else "%.7g"
                np.savetxt(fh, rows, fmt=fmt)
            else:
                # legacy VTK binary is big-endian, always
                big = payload.astype(payload.dtype.newbyteorder(">"))
                fh.write(big.tobytes())
                fh.write(b"\n")
    return path


def write_vtk(path, field, name="velocity", kind=None, solid=None,
              magnitude=True, ascii_mode=False, tile=1, title="perm-tensor-topology"):
    """One named field to `path`: the single-array case of `write_vtk_multi`."""
    return write_vtk_multi(path, {name: field}, solid=solid, magnitude=magnitude,
                           ascii_mode=ascii_mode, tile=tile,
                           kinds={name: kind} if kind else None, title=title)


# ---------------------------------------------------------------------------


def _self_test():
    """
    The checks that matter are about ordering, not about VTK syntax: a file that
    parses but is transposed is the failure mode this tool can actually have.
    """
    import struct
    import tempfile
    from pathlib import Path

    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}")

    tmp = Path(tempfile.mkdtemp())

    # 1. a field that is a known function of (x, y) must come back with x fastest
    ny, nx = 3, 5
    yy, xx = np.mgrid[0:ny, 0:nx]
    f = np.stack([xx, yy], axis=-1).astype(np.float64)     # v = (x, y)
    write_vtk(tmp / "v.vtk", f, ascii_mode=True, magnitude=False)
    text = (tmp / "v.vtk").read_text().splitlines()
    i = text.index("VECTORS velocity float")
    body = np.array([[float(t) for t in ln.split()] for ln in text[i + 1: i + 1 + nx * ny]])
    check("DIMENSIONS is (nx, ny, 1)", "DIMENSIONS 5 3 1" in text)
    check("first row is x=0..nx-1 at y=0 (x varies fastest)",
          np.array_equal(body[:nx, 0], np.arange(nx)) and np.all(body[:nx, 1] == 0))
    check("second row is y=1", np.all(body[nx:2 * nx, 1] == 1))
    check("2-component input padded with vz=0", np.all(body[:, 2] == 0))

    # 2. binary round-trip, big-endian, correct byte count
    write_vtk(tmp / "v.bin.vtk", f, magnitude=False)
    raw = (tmp / "v.bin.vtk").read_bytes()
    payload = raw.split(b"VECTORS velocity float\n", 1)[1][:-1]
    check("binary payload is npoints*3 float32", len(payload) == nx * ny * 3 * 4)
    vals = struct.unpack(f">{nx*ny*3}f", payload)
    check("binary matches ascii", np.allclose(np.array(vals).reshape(-1, 3), body))

    # 3. a 3D field keeps (z, y, x) ordering
    nz3, ny3, nx3 = 2, 3, 4
    zz, yy3, xx3 = np.mgrid[0:nz3, 0:ny3, 0:nx3]
    g = np.stack([xx3, yy3, zz], axis=-1).astype(np.float64)
    write_vtk(tmp / "g.vtk", g, ascii_mode=True, magnitude=False)
    t3 = (tmp / "g.vtk").read_text().splitlines()
    j = t3.index("VECTORS velocity float")
    b3 = np.array([[float(t) for t in ln.split()] for ln in t3[j + 1: j + 1 + nx3 * ny3 * nz3]])
    check("3D DIMENSIONS is (nx, ny, nz)", "DIMENSIONS 4 3 2" in t3)
    check("3D order is x fastest then y then z",
          np.array_equal(b3, g.reshape(-1, 3)))

    # 4. scalar + solid companion array
    solid = (xx + yy) % 2
    write_vtk(tmp / "s.vtk", f, solid=solid, ascii_mode=True)
    st = (tmp / "s.vtk").read_text()
    check("magnitude scalar written", "SCALARS velocity_magnitude float 1" in st)
    check("solid scalar written", "SCALARS solid unsigned_char 1" in st)
    check("one POINT_DATA declaration only", st.count("POINT_DATA") == 1)

    # 5. shape guards
    try:
        write_vtk(tmp / "bad.vtk", f, solid=np.zeros((2, 2)), ascii_mode=True)
        check("mismatched solid rejected", False)
    except ValueError:
        check("mismatched solid rejected", True)

    check("scalar (ny,nx) auto-detected", _guess_kind(np.zeros((4, 4))) == "scalar")
    check("vector (ny,nx,2) auto-detected", _guess_kind(np.zeros((4, 4, 2))) == "vector")

    # 6. several named fields in one file, the form solve_velocity3d.py writes
    g2 = np.stack([yy, xx], axis=-1).astype(np.float64)          # v = (y, x)
    write_vtk_multi(tmp / "m.vtk", {"velocity_fx": f, "velocity_fy": g2},
                    solid=solid, ascii_mode=True)
    mt = (tmp / "m.vtk").read_text()
    for want in ("VECTORS velocity_fx float", "VECTORS velocity_fy float",
                 "SCALARS velocity_fx_magnitude float 1",
                 "SCALARS velocity_fy_magnitude float 1",
                 "SCALARS solid unsigned_char 1"):
        check(f"multi: {want!r} present", want in mt)
    check("multi: one POINT_DATA declaration only", mt.count("POINT_DATA") == 1)
    check("multi: solid written once", mt.count("SCALARS solid") == 1)

    # each field keeps its own values (a shared buffer would alias them)
    ml = mt.splitlines()
    for nm, src in (("velocity_fx", f), ("velocity_fy", g2)):
        k = ml.index(f"VECTORS {nm} float")
        body = np.array([[float(s) for s in ln.split()]
                         for ln in ml[k + 1: k + 1 + nx * ny]])
        check(f"multi: {nm} holds its own data",
              np.array_equal(body[:, :2], src.reshape(-1, 2)))

    try:
        write_vtk_multi(tmp / "bad2.vtk", {"a": f, "b": np.zeros((2, 2, 2))},
                        ascii_mode=True)
        check("multi: mismatched grids rejected", False)
    except ValueError:
        check("multi: mismatched grids rejected", True)

    try:
        write_vtk_multi(tmp / "bad3.vtk", {"two words": f}, ascii_mode=True)
        check("multi: whitespace in array name rejected", False)
    except ValueError:
        check("multi: whitespace in array name rejected", True)

    # 7. periodic tiling
    write_vtk_multi(tmp / "t.vtk", {"velocity": f}, solid=solid, tile=3,
                    ascii_mode=True, magnitude=False)
    tt = (tmp / "t.vtk").read_text().splitlines()
    check("tile=3 scales DIMENSIONS", f"DIMENSIONS {nx*3} {ny*3} 1" in tt)
    check("tile=3 scales POINT_DATA", f"POINT_DATA {nx*ny*9}" in tt)
    kk = tt.index("VECTORS velocity float")
    tb = np.array([[float(s) for s in ln.split()]
                   for ln in tt[kk + 1: kk + 1 + nx * ny * 9]]).reshape(ny*3, nx*3, 3)
    check("tile=3 block (1,1) equals block (0,0)",
          np.array_equal(tb[ny:2*ny, nx:2*nx], tb[:ny, :nx]))
    check("tile=3 first block is the original field",
          np.array_equal(tb[:ny, :nx, :2], f))
    check("tile=1 is a no-op", np.array_equal(tile_field(f, 1, "vector"), f))
    check("tile_field on a 3D vector keeps the component axis",
          tile_field(g, 2, "vector").shape == (nz3*2, ny3*2, nx3*2, 3))
    check("tile_field on a 3D scalar",
          tile_field(zz, 2, "scalar").shape == (nz3*2, ny3*2, nx3*2))
    try:
        tile_field(f, 0, "vector")
        check("tile < 1 rejected", False)
    except ValueError:
        check("tile < 1 rejected", True)

    # write_vtk must stay exactly the single-field case of write_vtk_multi
    write_vtk(tmp / "one_a.vtk", f, ascii_mode=True, solid=solid)
    write_vtk_multi(tmp / "one_b.vtk", {"velocity": f}, ascii_mode=True, solid=solid)
    check("write_vtk == write_vtk_multi on one field",
          (tmp / "one_a.vtk").read_bytes() == (tmp / "one_b.vtk").read_bytes())

    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input .npy")
    p.add_argument("output", nargs="?", help="output .vtk")
    p.add_argument("-o", "--output-file", dest="out",
                   help="output .vtk (required with --field)")
    p.add_argument("--field", action="append", default=[], metavar="NAME=FILE",
                   help="add a named array; repeatable, all on one grid")
    p.add_argument("--name", default="velocity", help="array name in the VTK file")
    p.add_argument("--kind", choices=["vector", "scalar"],
                   help="override the vector/scalar guess")
    p.add_argument("--solid", help="optional .npy mask on the same grid")
    p.add_argument("--tile", type=int, default=1, metavar="N",
                   help="replicate the periodic cell N times along each axis so "
                        "streamlines cross the seams (costs N^2 in 2D, N^3 in 3D)")
    p.add_argument("--ascii", action="store_true", help="ASCII instead of binary")
    p.add_argument("--no-magnitude", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        return _self_test()

    solid = np.load(a.solid) if a.solid else None
    mode = "ascii" if a.ascii else "binary"

    if a.field:
        if a.input or a.output:
            p.error("use --field with -o/--output-file, not positional arguments")
        out = a.out or p.error("-o/--output-file is required with --field")
        fields = {}
        for spec in a.field:
            if "=" not in spec:
                p.error(f"--field wants NAME=FILE, got {spec!r}")
            name, _, src = spec.partition("=")
            fields[name] = np.load(src)
            print(f"  {name:20s} {src} {fields[name].shape}")
        write_vtk_multi(out, fields, solid=solid, tile=a.tile,
                        magnitude=not a.no_magnitude, ascii_mode=a.ascii)
        print(f"{len(fields)} fields -> {out} ({mode})")
        return 0

    out = a.output or a.out
    if not a.input or not out:
        p.error("input and output are required (or use --field, or --self-test)")
    field = np.load(a.input)
    write_vtk(out, field, name=a.name, kind=a.kind, solid=solid, tile=a.tile,
              magnitude=not a.no_magnitude, ascii_mode=a.ascii)
    print(f"{a.input} {field.shape} -> {out} ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
