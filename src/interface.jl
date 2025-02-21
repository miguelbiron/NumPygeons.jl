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
    A NumPyro MCMC kernel linked to the model of interest.
    """
    kernel::Py
end

"""
$SIGNATURES

Create a [`NumPyroPath`](@ref) from model arguments. Note: following the 
NumPyro convention, the model itself should be passed to the kernel inside
the explorer.
"""
function NumPyroPath(;model_args::Py = pytuple(()), model_kwargs::Py = pydict())
    @assert is_python_tuple(model_args) "`model_args` should be a python tuple."
    @assert is_python_dict(model_kwargs) "`model_args` should be a python dict."
    
    # put placeholders in the rest of the fields; resolve in `create_path`
    NumPyroPath(
        model_args, model_kwargs, PythonCall.pynew(), PythonCall.pynew()
    )
end

"""
$SIGNATURES

Defines a tempered version of a NumPyro model.

$FIELDS
"""
struct NumPyroLogPotential
    """
    The numpyro MCMC kernel associated with the model at hand.
    """
    kernel::Py

    """
    A python object representing a tempered potential function. 
    Note: Following the NumPyro convention, a potential is the negative of
    a Pigeons logpotential.    
    """
    potential_fn::Py
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

A Pigeons explorer defined by a NumPyro MCMC kernel from the 
[`autostep`](https://github.com/UBC-Stat-ML/autostep) python package.

$FIELDS
"""
struct NumPyroExplorer
    """
    An autostep kernel.
    """
    kernel::Py

    """
    Number of times that the
    """
    n_refresh::Py
end

NumPyroExplorer(kernel, n_refresh::Int = 3) = 
    NumPyroExplorer(kernel, pyint(n_refresh))


###############################################################################
# PT initialization
###############################################################################

# Update the fields in the path using the explorer
function Pigeons.create_path(
    path::NumPyroPath,
    inp::Inputs{NumPyroPath,<:Any,NumPyroExplorer}
    )
    # check we have a valid NumPyro MCMC kernel
    # note: kernel is later initialized in `Pigeons.initialization`
    kernel = inp.explorer.kernel
    @assert kernel isa Py && pyisinstance(kernel, numpyro.infer.mcmc.MCMCKernel)

    # make interpolator function
    interpolator = bridge.make_interpolator(
        kernel.model, path.model_args, path.model_kwargs
    )

    # update path fields and return
    PythonCall.pycopy!(path.interpolator, interpolator)
    PythonCall.pycopy!(path.kernel, kernel)
    return path
end

# Handle any other explorer type
Pigeons.create_path(::NumPyroPath, ::Inputs) = throw(ArgumentError(
    "Found incompatible explorer. Use a `NumPyroExplorer`."
))

# This won't ever be reached (error will pop-up in the previous function),
# but just in case
Pigeons.default_explorer(::NumPyroPath) = throw(ArgumentError(
    "No explorer found. Use a `NumPyroExplorer`."
))

# state initialization
# note: this occurs *after* the `Pigeons.Shared` object is created, and therefore
# after `Pigeons.create_path` is called.
function Pigeons.initialization(
    inp::Inputs{NumPyroPath,<:Any,NumPyroExplorer},
    replica_rng, 
    replica_idx
    )
    path = inp.target
    kernel = inp.explorer.kernel
    initial_kernel_state = kernel.init(
        jax_rng_key(replica_rng), 
        pyint(0), 
        pybuiltins.None, 
        path.model_args, 
        path.model_kwargs
    )
    return NumPyroState(initial_kernel_state)
end

###############################################################################
# log potential methods
###############################################################################

Pigeons.interpolate(path::NumPyroPath, beta::Real) = 
    NumPyroLogPotential(path.kernel, path.interpolator(pyfloat(beta)))

function (log_potential::NumPyroLogPotential)(state::NumPyroState)
    kernel = log_potential.kernel
    potential_fn = log_potential.potential_fn
    kernel_state = state.kernel_state
    unconstrained_sample = pygetattr(kernel_state, kernel.sample_field)
    return -pyconvert(Float64, pyfloat(potential_fn(unconstrained_sample))) # note: `pyfloat` converts the singleton JAX array to a python float
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
