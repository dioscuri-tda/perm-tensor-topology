#!/usr/bin/env python
"""Pack / unpack the binary structures of a dataset for distribution.

The structures are 80^3 voxel grids stored as `.raw` files (12-byte header +
one byte per voxel, see structure_io3d.py).  For the repository they are
bit-packed and compressed into a few `.npz` archives of 1000 structures each
(GitHub limits single files to 100 MB).  Unpacking writes byte-identical
`.raw` files and checks them against the SHA-256 stored at packing time.

    # unpack the dataset shipped in data/aniso3d/ to DATA/aniso3d/structures/
    python scripts/pack_structures.py unpack --packed data/aniso3d --out DATA/aniso3d

    # (maintainers) pack DATA/aniso3d/structures/ into data/aniso3d/
    python scripts/pack_structures.py pack --dataset DATA/aniso3d --packed data/aniso3d
"""
import argparse
import hashlib
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io3d import load_structure, save_structure  # noqa: E402

CHUNK = 1000


def pack(dataset, packed):
    meta = pd.read_csv(os.path.join(dataset, "structures.csv"))
    os.makedirs(packed, exist_ok=True)
    digests = []
    for c0 in range(0, len(meta), CHUNK):
        rows = meta.iloc[c0:c0 + CHUNK]
        bits, shapes = [], []
        for fn in rows.filename:
            path = os.path.join(dataset, "structures", fn)
            a = load_structure(path)
            shapes.append(a.shape)
            bits.append(np.packbits(a.ravel()))
            digests.append(hashlib.sha256(open(path, "rb").read()).hexdigest())
        assert len(set(shapes)) == 1
        out = os.path.join(packed, f"structures_{c0:05d}_{c0 + len(rows) - 1:05d}.npz")
        np.savez_compressed(out, bits=np.stack(bits), shape=np.array(shapes[0]),
                            sample_id=rows.sample_id.to_numpy(), filename=rows.filename.to_numpy())
        print(f"{out}  {os.path.getsize(out) / 2**20:.1f} MB")
    meta.assign(sha256=digests).to_csv(os.path.join(packed, "structures.csv"), index=False)


def unpack(packed, out):
    meta = pd.read_csv(os.path.join(packed, "structures.csv"))
    sha = dict(zip(meta.filename, meta.sha256))
    os.makedirs(os.path.join(out, "structures"), exist_ok=True)
    n = 0
    for f in sorted(x for x in os.listdir(packed) if x.startswith("structures_") and x.endswith(".npz")):
        z = np.load(os.path.join(packed, f), allow_pickle=True)
        shape = tuple(int(v) for v in z["shape"])
        for b, fn in zip(z["bits"], z["filename"]):
            a = np.unpackbits(b)[: np.prod(shape)].reshape(shape)
            path = os.path.join(out, "structures", fn)
            save_structure(path, a)
            if hashlib.sha256(open(path, "rb").read()).hexdigest() != sha[fn]:
                raise SystemExit(f"checksum mismatch: {path}")
            n += 1
    if n != len(meta):
        raise SystemExit(f"unpacked {n} structures, expected {len(meta)}")
    shutil.copy(os.path.join(packed, "structures.csv"), os.path.join(out, "structures.csv"))
    for extra in ("permeability.csv",):
        src = os.path.join(packed, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, extra))
    print(f"unpacked {n} structures to {out}/structures (all checksums OK)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack"); a.add_argument("--dataset", required=True); a.add_argument("--packed", required=True)
    b = sub.add_parser("unpack"); b.add_argument("--packed", required=True); b.add_argument("--out", required=True)
    args = p.parse_args()
    pack(args.dataset, args.packed) if args.cmd == "pack" else unpack(args.packed, args.out)
