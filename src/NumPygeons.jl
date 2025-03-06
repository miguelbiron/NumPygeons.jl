module NumPygeons

# PythonCall relies on `pickle` for serialization. But `pickle` does not 
# serialize closures correctly---`dill` does. See e.g.
# https://github.com/JuliaPy/PythonCall.jl/issues/424
ENV["JULIA_PYTHONCALL_PICKLE"] = "dill"

using DocStringExtensions
using Pigeons
using PythonCall
using SplittableRandoms: SplittableRandom

include("utils.jl")
include("NumPyroPath.jl")
include("recorders.jl")
include("interface.jl")

export NumPyroPath, 
    NumPyroExplorer,
    numpyro_trace

# import python packages
# this approach handles precompilation correctly
const autostep = PythonCall.pynew()
const jax = PythonCall.pynew()
const numpyro = PythonCall.pynew()
const bridge = PythonCall.pynew()
function __init__()
    ensure_bridge_exist()
    PythonCall.pycopy!(bridge, pyimport("numpygeons.bridge"))
    PythonCall.pycopy!(jax, pyimport("jax"))
    PythonCall.pycopy!(numpyro, pyimport("numpyro"))
    PythonCall.pycopy!(autostep, pyimport("autostep"))
    return
end


end # module NumPygeons
