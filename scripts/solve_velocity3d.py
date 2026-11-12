"""
Solve one 3D structure and keep the velocity fields, for visualisation.

Runs `lbm3d-perm` on one structure exactly as the dataset run does (three
solves of the same triply periodic cell, forced along +x, +y, +z, giving the
three columns of K) and additionally keeps the converged velocity field of each
solve.

Outputs, given `-o VIZ/v0`:

    VIZ/v0.fx.npy      (nz, ny, nx, 3) float32   velocity under x-forcing
    VIZ/v0.fy.npy      (nz, ny, nx, 3) float32   ... y-forcing
    VIZ/v0.fz.npy      (nz, ny, nx, 3) float32   ... z-forcing
    VIZ/v0.solid.npy   (nz, ny, nx)    uint8     1 = solid, 0 = pore
    VIZ/v0.csv                                   the solver's full tensor row
    VIZ/v0.vtk                                   with --vtk

The ParaView file is one dataset carrying all three directions (`velocity_fx`,
`velocity_fy`, `velocity_fz`, their magnitudes and `solid`, plus `velocity_val`
with --validate), so one pipeline serves all of them and the direction is
chosen from the array dropdown. `--split` writes one file per direction
instead. Threshold on `solid` to show the pore space.

The `.vel` format written by the solver (`--velocity`) is a 16-byte header of
four int32 (NX, NY, NZ, 3) followed by NN float32 triples, x varying fastest as
in the `.raw` structure, so a plain reshape to (nz, ny, nx, 3) is indexed
[z, y, x] like `structure_io3d.load_structure`. The exported velocity is the
physical Guo-forced velocity (sum f e + F/2)/rho.

Two consistency checks run automatically:

  * mask agreement: the solver never writes velocity at solid nodes, so the
    field must be exactly zero on every solid node;
  * flux reproduction: q_i = mean(u_i) over the whole cell (the Darcy
    superficial velocity) recomputed from the field must reproduce the
    solver's `qx_fx`... columns. The solver reaches a fixed point in double
    precision, so agreement should be at float32 precision (~1e-7).

Usage
-----
    python scripts/solve_velocity3d.py <structure.raw> -o VIZ/v0 --vtk
    python scripts/solve_velocity3d.py <structure.raw> -o VIZ/v0 --vtk --validate
"""

import argparse
import csv as csvmod
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_io3d import load_structure          # noqa: E402
from to_vtk import write_vtk, write_vtk_multi      # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOLVER = REPO / "lbm" / "lbm3d-perm"
VEL_HEADER_BYTES = 16


def read_velocity_vel(path, nz, ny, nx):
    """Parse one `--velocity` file into (nz, ny, nx, 3) float32."""
    raw = Path(path).read_bytes()
    if len(raw) < VEL_HEADER_BYTES:
        raise ValueError(f"{path}: too short to hold a header")
    hx, hy, hz, ncomp = np.frombuffer(raw, dtype="<i4", count=4)
    if (int(hx), int(hy), int(hz), int(ncomp)) != (nx, ny, nz, 3):
        raise ValueError(
            f"{path}: header says {hx}x{hy}x{hz}x{ncomp}, "
            f"structure is {nx}x{ny}x{nz}x3")
    expect = VEL_HEADER_BYTES + nx * ny * nz * 3 * 4
    if len(raw) != expect:
        raise ValueError(f"{path}: expected {expect} bytes, file has {len(raw)}")
    arr = np.frombuffer(raw, dtype="<f4", offset=VEL_HEADER_BYTES)
    return arr.reshape(nz, ny, nx, 3)


def volume_flux(field):
    """q_i = mean(u_i) over the whole cell, reproducing lbm3d-perm: volumeavg()."""
    return tuple(float(field[..., c].mean()) for c in range(3))


def run(args):
    structure = Path(args.structure)
    if not structure.exists():
        sys.exit(f"no such structure: {structure}")
    solver = Path(args.solver)
    if not solver.exists():
        sys.exit(f"solver not found: {solver}\n"
                 f"build it with: bash lbm/make_perm.sh")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    if csv.exists():
        csv.unlink()        # the solver appends; a stale row would be confusing

    cmd = [str(solver), str(structure), str(csv), "--velocity", str(out),
           "--ff", str(args.ff)]
    for flag, val in (("--eps", args.eps), ("--max-steps", args.max_steps),
                      ("--avg-steps", args.avg_steps), ("--lag", args.lag)):
        if val is not None:
            cmd += [flag, str(val)]
    if args.validate:
        cmd.append("--validate")
    if args.quiet:
        cmd.append("--quiet")

    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc == 2:
        sys.exit("solver failed (exit 2)")
    if rc == 3:
        print("\n!! solver exit 3: at least one direction did not converge; "
              "fields are still written but check conv_fx/conv_fy/conv_fz "
              "in the CSV\n")

    solid = load_structure(structure)
    nz, ny, nx = solid.shape

    tags = ["fx", "fy", "fz"] + (["val"] if args.validate else [])
    fields = {}
    for tag in tags:
        vel = Path(f"{out}.{tag}.vel")
        if not vel.exists():
            sys.exit(f"solver did not write {vel}")
        fields[tag] = read_velocity_vel(vel, nz, ny, nx)
        np.save(f"{out}.{tag}.npy", fields[tag])
        if not args.keep_vel:
            vel.unlink()
    np.save(f"{out}.solid.npy", solid)

    _report(fields, solid, csv, out)

    if args.vtk:
        print()
        if args.split:
            for tag, field in fields.items():
                force = "(1,1,1)" if tag == "val" else "+" + tag[-1]
                write_vtk(f"{out}.{tag}.vtk", field, name="velocity", solid=solid,
                          ascii_mode=args.ascii, tile=args.tile,
                          title=f"{structure.name} force={force}")
                mb = Path(f"{out}.{tag}.vtk").stat().st_size / 1e6
                print(f"  wrote {out}.{tag}.vtk  ({mb:.1f} MB)")
        else:
            write_vtk_multi(f"{out}.vtk",
                            {f"velocity_{tag}": field for tag, field in fields.items()},
                            solid=solid, ascii_mode=args.ascii, tile=args.tile,
                            title=f"{structure.name} force=+x,+y,+z")
            mb = Path(f"{out}.vtk").stat().st_size / 1e6
            print(f"  wrote {out}.vtk  ({mb:.1f} MB, arrays "
                  f"{', '.join('velocity_' + tg for tg in fields)}, solid)")

    return 0


def _report(fields, solid, csv, out):
    nz, ny, nx = solid.shape
    print(f"\nstructure {nx}x{ny}x{nz}, porosity {1.0 - solid.mean():.4f}")
    for tag, field in fields.items():
        print(f"  {out}.{tag}.npy  {field.shape} float32")

    print("\ncheck 1 - velocity is exactly zero on solid nodes")
    mask = solid.astype(bool)
    for tag, field in fields.items():
        n_bad = int((mask[..., None] & (field != 0.0)).any(-1).sum())
        worst = float(np.abs(field[mask]).max()) if mask.any() else 0.0
        print(f"  {tag}: {n_bad} solid nodes with nonzero velocity "
              f"(max |u| there {worst:.3e})  [{'ok' if n_bad == 0 else 'FAIL'}]")

    row = None
    if csv.exists():
        with open(csv) as fh:
            rows = list(csvmod.DictReader(fh))
        row = rows[-1] if rows else None
    if row is None:
        print("\ncheck 2 - skipped, no CSV row")
        return

    print("\ncheck 2 - flux recomputed from the field vs the solver's block mean")
    print(f"  {'':10s} {'from field':>14s} {'from CSV':>14s} {'rel diff':>10s}")
    for tag in ("fx", "fy", "fz"):
        if tag not in fields:
            continue
        q = volume_flux(fields[tag])
        for comp, mine in zip(("qx", "qy", "qz"), q):
            theirs = float(row[f"{comp}_{tag}"])
            denom = max(abs(theirs), 1e-30)
            print(f"  {comp}_{tag:3s} {mine:14.7e} {theirs:14.7e} "
                  f"{abs(mine - theirs)/denom:10.2e}")

    K = [[float(row[f"K_{r}{c}"]) for c in "xyz"] for r in "xyz"]
    print("\n  K =")
    for r in range(3):
        print("    [ " + "  ".join(f"{K[r][c]:12.6f}" for c in range(3)) + " ]")
    print(f"  recip_resid {float(row['recip_resid']):.2e}   "
          f"k1 {float(row['k1']):.4f}  k2 {float(row['k2']):.4f}  "
          f"k3 {float(row['k3']):.4f}  FA {float(row['fa']):.4f}")
    print(f"  full row: {csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure", help="structure .raw")
    p.add_argument("-o", "--output", required=True, help="output prefix, e.g. VIZ/v0")
    p.add_argument("--solver", default=str(DEFAULT_SOLVER))
    p.add_argument("--ff", type=float, default=1)
    p.add_argument("--eps", type=float)
    p.add_argument("--lag", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--avg-steps", type=int)
    p.add_argument("--validate", action="store_true",
                   help="also solve along (1,1,1) and keep that field too")
    p.add_argument("--vtk", action="store_true", help="also write a ParaView .vtk file")
    p.add_argument("--split", action="store_true",
                   help="one .vtk per forcing direction instead of one combined file")
    p.add_argument("--tile", type=int, default=1, metavar="N",
                   help="replicate the periodic cell N times along each axis in the "
                        ".vtk, so ParaView streamlines cross the seams instead of "
                        "stopping at them (costs N^3 in file size)")
    p.add_argument("--ascii", action="store_true",
                   help="ASCII VTK instead of binary (~15x larger; at 80^3 that "
                        "is ~90 MB per direction)")
    p.add_argument("--keep-vel", action="store_true",
                   help="keep the solver's raw binary .vel files")
    p.add_argument("--quiet", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
