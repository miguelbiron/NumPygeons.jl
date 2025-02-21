module NumPygeons

using DocStringExtensions
using Pigeons
using PythonCall

# import python packages
# this approach handles precompilation correctly
const autostep = PythonCall.pynew()
const jax = PythonCall.pynew()
const numpyro = PythonCall.pynew()
const bridge = PythonCall.pynew()
function __init__()
    PythonCall.pycopy!(bridge, pyimport("numpygeons.bridge"))
    PythonCall.pycopy!(autostep, pyimport("autostep"))
    PythonCall.pycopy!(jax, pyimport("jax"))
    PythonCall.pycopy!(numpyro, pyimport("numpyro"))
    return
end

"""
$SIGNATURES

Make a JAX random number generator (RNG) key from a Julia RNG.
"""
jax_rng_key(rng) = jax.random.key(pyint(Pigeons.java_seed(rng)))


include("utils.jl")

export NumPyroPath, NumPyroExplorer
include("interface.jl")


end # module NumPygeons
