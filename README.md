# perm-tensor-topology

Code and data for the paper

> **Permeability tensor of anisotropic porous media from directional pore-space topology**
> (authors, journal and DOI to be added)

We predict the full 3×3 permeability tensor of three-dimensional porous structures
from their geometry. The tensors are computed with a lattice Boltzmann solver. The
structures are described by two direction-aware topological descriptors:
- **TDA**, a cone-based filtration;
- **THR**, a throat-scale filtration built on the distance transform of the pore space.

Both are compared with statistical microstructure descriptors and with physics-based
estimators, using gradient-boosted trees (CatBoost).

## Contents

| path | what |
|---|---|
| `data/aniso3d/` | the synthetic dataset: 5000 periodic anisotropic structures (80³ voxels, bit-packed, ~100 MB) and their permeability tensors (`permeability.csv`) |
| `splits/` | the frozen hold-out set of the learning curves |
| `lbm/` | the lattice Boltzmann solver (C++, D3Q19, BGK, periodic, body force) |
| `scripts/` | generator, solver driver, descriptors, training and analysis |
| `pipeline/` | the exact commands that produced the results of the paper, step by step |
| `julia/` | Julia project for the Euler characteristic profiles (`setup.jl` installs it) |

## Installation

```bash
conda env create -f environment.yml
conda activate perm-tensor-topology
```

- **Euler characteristic profiles (part of TDA):** computed in Julia (≥ 1.10) with the
  `Ecp` package from [ECP_toolkit](https://github.com/Malinon/ECP_toolkit). It is not in
  the General registry; `julia --project=julia julia/setup.jl` installs it at the commit
  used for the paper. To use another Julia project that has it, set `JLENV`.
- **LBM solver:** needs only `g++` (`bash lbm/make_perm.sh`).

## Reproducing the paper

Run all commands from the repository root. Every step is resumable.

| step | command | output | cost (order of magnitude) |
|---|---|---|---|
| 1. structures | `bash pipeline/01_data.sh` | `DATA/aniso3d/` | seconds (unpack) |
| 2. permeability (optional) | `bash pipeline/02_lbm.sh` | `DATA/aniso3d/permeability.csv` | ~3 CPU-hours per structure; the shipped tensors are copied by step 1 |
| 3. descriptors | `bash pipeline/03_descriptors.sh aniso3d` | `DESC/aniso3d/{baselines,phys,cone,throat}.csv` | ~10 h on 24 cores |
| 4. models | `bash pipeline/04_train.sh aniso3d` | `RESULTS/aniso3d/seed{0,1,2}/` | ~1 day on 50 cores |
| 5. learning curves | `bash pipeline/05_learning_curve.sh` | `RESULTS/aniso3d/lc_*/` | ~10 h on 52 cores |
| 6. SHAP | `bash pipeline/06_shap.sh` | `RESULTS/aniso3d/shap/` | ~1 h |

`scripts/solve_velocity3d.py` writes the velocity fields of a structure as VTK files
for visualisation (e.g. in ParaView).

## The dataset

`data/aniso3d/permeability.csv` holds one row per structure. The main columns are:
- `sample_id`, `filename`;
- `stratum`: the anisotropy class, `iso`, `mod` or `strong`;
- `aniso_A`, `aniso_B`, `e1_*`, `e2_*`, `k0`, `n_modes`: the parameters of the generator;
- `porosity`;
- `K_ij`: the raw tensor in lattice units, with column j taken from the solve forced along j;
- `k_xy`, `k_xz`, `k_yz`: the symmetrised off-diagonal components;
- `k1..k3`, `v1_*..v3_*`: the principal permeabilities and axes;
- `steps_*`, `conv_*`: the convergence record of the three solves.

The targets used in the paper are `K_xx, K_yy, K_zz` and `k_xy, k_xz, k_yz`.

The structures are stored as `.raw` files after unpacking: a 12-byte header
(`nx, ny, nz` as little-endian int32) followed by one byte per voxel, 1 = solid,
0 = pore, in (z, y, x) order (`scripts/structure_io3d.py`). They are triply periodic.
`scripts/pack_structures.py` converts between the packed and the unpacked form and
checks every file against its SHA-256.

## Descriptor sets: paper names and code names

The code and the `RESULTS/` directories use the group keys of `scripts/train_catboost.py`:

| paper | code group | columns |
|---|---|---|
| Stats | `baselines` | `por_`, `tpc_`, `fab_`, `fabdir_` |
| Phys | `phys` | `kc_`, `cp_` |
| Baselines (Stats + Phys) | `classical` | |
| TDA | `tda` | `ecp_`, `ph_` (cone filtration) |
| THR | `thr` | `thr_` (throat filtration) |
| Stats + TDA / Stats + THR | `all` / `baselines+thr` | |
| Phys + TDA / Phys + THR | `phys+tda` / `phys+thr` | |
| Baselines + TDA / Baselines + THR | `classical+tda` / `classical+thr` | |

## Citation

To be added on publication.

## License

Code: MIT (`LICENSE`). Data in `data/`: CC BY 4.0 (`data/LICENSE`).
