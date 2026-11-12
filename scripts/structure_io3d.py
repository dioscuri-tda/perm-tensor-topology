"""
Read and write 3D binary porous structures in the `.raw` format.

This module is the single place where Python reads or writes a 3D structure.
The same bytes feed the lattice Boltzmann solver (`lbm/lbm3d-perm.cpp`) and the
topological descriptors, so there is no second copy that could drift out of
sync.

Format
------
    bytes  0..11   three int32, little-endian: nx, ny, nz
    bytes 12..     nx*ny*nz uint8, 1 = solid, 0 = pore

Voxels are stored in C order over ``[z][y][x]``, i.e. x varies fastest. This
is the order VTK STRUCTURED_POINTS expects (`to_vtk.py`) and the natural
reading order of the C++ solver, so no transpose is needed anywhere. The array
returned here is indexed ``[z, y, x]`` = ``[solver_z, solver_y, solver_x]``.

The 12-byte header makes the file self-describing: a size mismatch is caught on
load instead of silently reinterpreting the voxels.

Usage
-----
    >>> from structure_io3d import load_structure, save_structure
    >>> s = load_structure("DATA/aniso3d/structures/sample_000000_....raw")
    >>> s.shape, s.dtype, s.max()
    ((80, 80, 80), dtype('uint8'), 1)
    >>> porosity = 1.0 - s.mean()

    python scripts/structure_io3d.py      # runs the self-test
"""

from pathlib import Path

import numpy as np

__all__ = ["load_structure", "save_structure", "porosity", "MAGIC_HEADER_BYTES"]

MAGIC_HEADER_BYTES = 12


def save_structure(path, solid) -> Path:
    """Write a (nz, ny, nx) array of {0,1} as .raw. 1 = solid."""
    solid = np.ascontiguousarray(solid).astype(np.uint8)
    if solid.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {solid.shape}")
    if solid.max() > 1:
        raise ValueError("structure must be binary (0 = pore, 1 = solid)")
    nz, ny, nx = solid.shape
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(np.array([nx, ny, nz], dtype="<i4").tobytes())
        fh.write(solid.tobytes())
    return path


def load_structure(path) -> np.ndarray:
    """Read a .raw structure as uint8 (nz, ny, nx), indexed [z, y, x]. 1 = solid."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < MAGIC_HEADER_BYTES:
        raise ValueError(f"{path}: too short to hold a header")
    nx, ny, nz = np.frombuffer(raw, dtype="<i4", count=3)
    expect = MAGIC_HEADER_BYTES + int(nx) * int(ny) * int(nz)
    if len(raw) != expect:
        raise ValueError(
            f"{path}: header says {nx}x{ny}x{nz} -> {expect} bytes, file has {len(raw)}")
    arr = np.frombuffer(raw, dtype=np.uint8, offset=MAGIC_HEADER_BYTES)
    return arr.reshape(int(nz), int(ny), int(nx))


def porosity(solid) -> float:
    """Pore fraction, matching the `porosity` column of structures.csv."""
    return float(1.0 - np.asarray(solid).mean())


def _self_test() -> int:
    import tempfile

    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}")

    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)

    # non-cubic on purpose: a bug that swaps two equal axes would go unnoticed
    a = (rng.random((5, 7, 11)) > 0.5).astype(np.uint8)
    save_structure(tmp / "a.raw", a)
    b = load_structure(tmp / "a.raw")
    check("round-trip preserves values", np.array_equal(a, b))
    check("round-trip preserves (nz, ny, nx) shape", b.shape == (5, 7, 11))

    size = MAGIC_HEADER_BYTES + 5 * 7 * 11
    check(f"file is 12-byte header + nx*ny*nz bytes ({size})",
          (tmp / "a.raw").stat().st_size == size)

    hdr = np.frombuffer((tmp / "a.raw").read_bytes(), dtype="<i4", count=3)
    check("header stores (nx, ny, nz) = (11, 7, 5)", list(hdr) == [11, 7, 5])

    # x fastest: voxel (z,y,x) must sit at ((z*ny)+y)*nx+x in the payload
    payload = np.frombuffer((tmp / "a.raw").read_bytes(), dtype=np.uint8,
                            offset=MAGIC_HEADER_BYTES)
    z, y, x = 3, 4, 9
    check("x varies fastest in the payload",
          payload[(z * 7 + y) * 11 + x] == a[z, y, x])

    truncated = (tmp / "a.raw").read_bytes()[:-3]
    (tmp / "bad.raw").write_bytes(truncated)
    try:
        load_structure(tmp / "bad.raw")
        check("truncated file rejected", False)
    except ValueError:
        check("truncated file rejected", True)

    try:
        save_structure(tmp / "c.raw", np.full((2, 2, 2), 7))
        check("non-binary input rejected", False)
    except ValueError:
        check("non-binary input rejected", True)

    check("porosity is the pore fraction",
          abs(porosity(np.array([[[1, 0], [0, 0]]])) - 0.75) < 1e-12)

    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_self_test())
