module NumPygeons

using DocStringExtensions
using Pigeons
using PythonCall
using SplittableRandoms: SplittableRandom

# import python packages
# this approach handles precompilation correctly
const autostep = PythonCall.pynew()
const jax = PythonCall.pynew()
const numpyro = PythonCall.pynew()
const bridge = PythonCall.pynew()
function __init__()
    PythonCall.pycopy!(autostep, pyimport("autostep"))
    PythonCall.pycopy!(jax, pyimport("jax"))
    PythonCall.pycopy!(numpyro, pyimport("numpyro"))
    PythonCall.pycopy!(bridge, pyimport("numpygeons.bridge"))
    return
end

include("utils.jl")
include("NumPyroPath.jl")
include("recorders.jl")
include("interface.jl")

export NumPyroPath, 
    NumPyroExplorer,
    numpyro_trace




end # module NumPygeons
