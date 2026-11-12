# Installs the Julia dependencies of scripts/ecp3d.jl into this project:
#   julia --project=julia julia/setup.jl
# Ecp (Euler characteristic profiles) is not in the General registry; it is
# installed from GitHub at the commit used for the paper.
import Pkg
Pkg.add(url="https://github.com/Malinon/ECP_toolkit", rev="d110d1ebcb7078736468fdf9eee05328b6bc4691")
Pkg.add(["NPZ", "Glob"])
Pkg.instantiate()
