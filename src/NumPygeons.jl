module NumPygeons

using Pigeons
using PythonCall

# initialization (handles precompilation correctly)
# const autostep = PythonCall.pynew()
# const bridge = PythonCall.pynew()
# function __init__()
#     PythonCall.pycopy!(bridge, pyimport("autostep").bridge)
#     PythonCall.pycopy!(bridge, pyimport("numpygeons").bridge)
# end

export NumPyroPath
include("interface.jl")

end # module NumPygeons
