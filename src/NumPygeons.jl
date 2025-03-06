module NumPygeons

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
