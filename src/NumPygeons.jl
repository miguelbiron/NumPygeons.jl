module NumPygeons

using Pigeons
using PythonCall

# initialization (handles precompilation correctly)
const autostep = PythonCall.pynew()
const numpyro = PythonCall.pynew()
const bridge = PythonCall.pynew()
function __init__()
    PythonCall.pycopy!(autostep, pyimport("autostep"))
    PythonCall.pycopy!(numpyro, pyimport("numpyro"))
    PythonCall.pycopy!(bridge, pyimport("numpygeons").bridge)
end

export NumPyroPath
include("interface.jl")

end # module NumPygeons
