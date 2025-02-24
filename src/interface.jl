# Approach for the Pigeons -- NumPyro interface
#
# Basic definitions
#   - replica.state is a full kernel state
#   - kernel is initialized with num_warmup=0 so that it never adapts within
#     kernel.sample. Note that it still keeps track of adaptation statistics!
#   - define a custom recorder that simply grabs the latest stats object in the state
# plan for recorder:
#   - implement merge in python
#   - no need to do anything with the recorder at end of round---it is constantly overwritten! 
# Plan for `step!`:
#   - call a python function that performs `kernel.sample` `n_refresh` times
#   - at the end of the call, we have a kernel state with which we update replica
#   - record at the end of each `step!`
# Plan for adaptation:
#   - In autostep, separate the non-trivial branch of `kernel.adapt`` into a function,
#     such that it can be directly called with a stats object.

###############################################################################
# type declarations and constructors
###############################################################################

"""
$SIGNATURES

Provides a linear prior-posterior path for NumPyro models.

$FIELDS
"""
struct NumPyroPath
    """
    A NumPyro MCMC kernel linked to the model of interest. This follows the 
    NumPyro convention that the model should be passed to this kernel's 
    constructor.
    """
    kernel::Py

    """
    A python tuple with optional arguments to the NumPyro model.
    """
    model_args::Py

    """
    A python dictionary with optional keyword arguments to the NumPyro model.
    """
    model_kwargs::Py

    """
    A python function that takes inverse temperatures and produces
    tempered potential functions compatible with NumPyro MCMC samplers.
    """
    interpolator::Py

    """
    An example of a kernel state that allows us to initialize other kernels
    targeting tempered versions of a NumPyro model in `Pigeons.interpolate`.
    """
    prototype_kernel_state::Py
end

"""
$SIGNATURES

Create a [`NumPyroPath`](@ref) from model arguments.
"""
function NumPyroPath(
    kernel::Py, 
    model_args::Py = pytuple(()), 
    model_kwargs::Py = pydict()
    )
    @assert is_python_tuple(model_args) "`model_args` should be a python tuple."
    @assert is_python_dict(model_kwargs) "`model_args` should be a python dict."
    
    # put placeholders in the rest of the fields; resolve in `create_path`
    NumPyroPath(
        kernel, model_args, model_kwargs, 
        PythonCall.pynew(), PythonCall.pynew()
    )
end

"""
$SIGNATURES

Defines a tempered version of a NumPyro model. Note: given the tight 
association in NumPyro between kernels and target potential functions, we use 
the Pigeons log_potential interface to carry local kernels that sample from 
tempered potentials.

$FIELDS
"""
struct NumPyroLogPotential{T<:Real}
    """
    A local `MCMCKernel` that targets a tempered version of the original model.
    """
    local_kernel::Py
end

"""
$SIGNATURES

Replica state for a NumPyro target.

$FIELDS
"""
struct NumPyroState
    """
    Full state of the MCMC kernel used within NumPyro.
    """
    kernel_state::Py
end

"""
$SIGNATURES

A Pigeons explorer defined by a NumPyro MCMC kernel.

$FIELDS
"""
struct NumPyroExplorer
    """
    Number of times that the
    """
    n_refresh::Py
end

NumPyroExplorer(;n_refresh::Int = 3) = NumPyroExplorer(pyint(n_refresh))

###############################################################################
# PT initialization
###############################################################################

function Pigeons.create_path(
    path::NumPyroPath, 
    ::Inputs{NumPyroPath,<:Any,NumPyroExplorer}
    )
    # check we have a valid NumPyro MCMC kernel
    kernel = path.kernel
    @assert kernel isa Py && pyisinstance(kernel, numpyro.infer.mcmc.MCMCKernel)

    # make interpolator function
    interpolator = bridge.make_interpolator(
        kernel.model, path.model_args, path.model_kwargs
    )

    # init the kernel
    kernel = inp.explorer.kernel
    prototype_kernel_state = kernel.init(
        jax_rng_key(SplittableRandom(inp.seed)), 
        pyint(0), 
        pybuiltins.None, 
        path.model_args, 
        path.model_kwargs
    )

    # update path fields and return
    PythonCall.pycopy!(path.interpolator, interpolator)
    PythonCall.pycopy!(path.prototype_kernel_state, prototype_kernel_state)
    return path
end

# Handle any other explorer type
Pigeons.create_path(::NumPyroPath, ::Inputs) = throw(ArgumentError(
    "Found incompatible explorer. Use a `NumPyroExplorer`."
))

# This will never be reached (the above exception will be thrown first) but 
# just for completeness 
Pigeons.default_explorer(::NumPyroPath) = NumPyroExplorer()

# state initialization
# note: this occurs *after* the `Pigeons.Shared` object is created, and therefore
# after `Pigeons.create_path` is called.
# TODO: use iid sampling when that is sorted out
Pigeons.initialization(
    inp::Inputs{NumPyroPath,<:Any,NumPyroExplorer},
    replica_rng, 
    replica_idx
    ) =
    NumPyroState(copymodule.deepcopy(inp.target.prototype_kernel_state))

###############################################################################
# log potential methods
###############################################################################

# Path interpolation
function Pigeons.interpolate(path::NumPyroPath, beta::Real)
    potential_fn = path.interpolator(pyfloat(beta))
    global_kernel = path.kernel
    local_kernel = pytype(global_kernel)(potential_fn = potential_fn)
    init_params = pygetattr(
        path.prototype_kernel_state, 
        global_kernel.sample_field
    )
    local_kernel.init(
        jax.random.split(SplittableRandom(inp.seed)), 
        pyint(0), 
        pybuiltins.None, 
        path.model_args, 
        path.model_kwargs
    )
    NumPyroLogPotential{typeof(beta)}(local_kernel)
end

function (log_potential::NumPyroLogPotential{T})(state::NumPyroState) where T
    kernel = log_potential.kernel
    potential_fn = log_potential.potential_fn
    kernel_state = state.kernel_state
    unconstrained_sample = pygetattr(kernel_state, kernel.sample_field)
    return -pyconvert(T, pyfloat(potential_fn(unconstrained_sample))) # note: `pyfloat` converts the singleton JAX array to a python float
end

###############################################################################
# sampling methods
###############################################################################

function Pigeons.step!(explorer::NumPyroExplorer, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    kernel_state = replica.state.kernel_state
    kernel = explorer.kernel
    path = shared.tempering.path

    # update the potential function of the kernel and sample `n_refresh` times
    kernel._potential_fn = log_potential.potential_fn
    new_kernel_state = bridge.loop_sample(
        kernel,
        kernel_state,
        explorer.n_refresh,
        path.model_args,
        path.model_kwargs
    )

    # update the replica state and return
    PythonCall.pycopy!(replica.state.kernel_state, new_kernel_state)

    # TODO: record adaptation stuff
    # if shared.iterators.scan == Pigeons.n_scans_in_round(shared.iterators)

    return    
end

# TODO: iid sampling from the prior
# function Pigeons.sample_iid!(log_potential::NumPyroLogPotential, replica, shared)
#     replica.state = Pigeons.initialization(log_potential, replica.rng, replica.replica_index)
# end

# TODO: explorer adaptation
# function Pigeons.adapt_explorer(
#     explorer::NumPyroExplorer, 
#     reduced_recorders, 
#     current_pt, 
#     new_tempering
#     )
# end
