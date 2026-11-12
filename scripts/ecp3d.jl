#=
Euler characteristic profiles (ECP) of the 3D cone filtration.

For every filtration `.npy` in <filt_dir> (a (Z, Y, X, n_params) float array
written by `filtration3d.py`: channels cone, dir_long, dir_norm), computes the
multi-parameter Euler characteristic profile with `Ecp.compute_contributions_3d`
and samples it on a regular grid of (grid_res + 1) points per filtration
parameter. With the three channels and the default --grid-res 6 this gives
(6+1)^3 = 343 values per structure and direction (--grid-res 8 gives 729).
The construction follows Ref. [Topolnicki et al. 2026].

n_params is read from the array, so a 2-channel filtration
(`filtration3d.py --channels 2`) also works. `vectorize_ecp` needs a tuple of
grid steps, one per filtration parameter.

Output: one flattened Float64 `.npy` per input file, same basename, in
<out_dir>; `collect_descriptors.py` turns these into columns `ecp_<tag>_<i>`.
Existing outputs are skipped, so an interrupted run can be restarted. A
structure whose ECP fails is reported and written as zeros.

Usage:
  julia --project=<env> --threads auto scripts/ecp3d.jl <filt_dir> <out_dir> [--grid-res 6]
=#

using Glob
using Base.Threads
import NPZ
import Ecp

function get_ecp(input_file, output_file, grid_res)
    data = NPZ.npzread(input_file)
    n_params = size(data, 4)
    try
        contr = Ecp.compute_contributions_3d(data)
        out = Ecp.vectorize_ecp(contr, Tuple(grid_res for _ in 1:n_params))
        NPZ.npzwrite(output_file, vec(Float64.(out)))
    catch e
        println("Error in ECP for $input_file: ", e)
        NPZ.npzwrite(output_file, fill(0.0, (grid_res + 1)^n_params))
    end
end

function parse_args()
    if length(ARGS) < 2
        println("usage: julia scripts/ecp3d.jl <filtration_dir> <output_dir> [--grid-res N]")
        exit(1)
    end
    grid_res = 6
    i = 3
    while i <= length(ARGS)
        if ARGS[i] == "--grid-res" && i + 1 <= length(ARGS)
            grid_res = parse(Int, ARGS[i + 1]); i += 2
        else
            println("unknown argument: $(ARGS[i])"); exit(1)
        end
    end
    return ARGS[1], ARGS[2], grid_res
end

const INPUT_DIR, OUTPUT_DIR, GRID_RES = parse_args()

mkpath(OUTPUT_DIR)
files = glob("*.npy", INPUT_DIR)
println("ECP: $(length(files)) files, grid-res $GRID_RES, $(nthreads()) threads")

done = Threads.Atomic{Int}(0)
@threads for i in eachindex(files)
    out = joinpath(OUTPUT_DIR, basename(files[i]))
    if !isfile(out)
        get_ecp(files[i], out, GRID_RES)
        Threads.atomic_add!(done, 1)
    end
end

println("ECP done: $(done[]) computed, $(length(files) - done[]) cached -> $OUTPUT_DIR")
