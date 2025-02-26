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
Pigeons.explorer_recorder_builders(::NumPyroExplorer) = [numpyro_trace]

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

    # init local kernel
    # note: rng_key is not actually reused because there is no sampling involved
    # when the kernel is not linked to a model. Moreover, the kernel_state 
    # returned by this function is discarded
    local_kernel.init(
        path.prototype_kernel_state.rng_key,  
        pyint(0), 
        init_params, 
        pybuiltins.None,
        pybuiltins.None
    )
    return NumPyroLogPotential{typeof(beta)}(local_kernel)
end

function (log_potential::NumPyroLogPotential{T})(state::NumPyroState) where T
    local_kernel = log_potential.local_kernel
    potential_fn = local_kernel._potential_fn
    kernel_state = state.kernel_state
    unconstrained_sample = pygetattr(kernel_state, local_kernel.sample_field)
    return -pyconvert(T, pyfloat(potential_fn(unconstrained_sample))) # note: `pyfloat` converts the singleton JAX array to a python float
end

###############################################################################
# sampling methods
###############################################################################

# iid sampling from the prior, for initialization and refreshment steps
function _sample_iid(path::NumPyroPath, kernel_state)
    unconstrained_sample = path.prior_sampler() # this function is initialized with its own RNG; see create_path
    return NumPyroState(
        bridge.update_sample_field(
            kernel_state, 
            path.kernel.sample_field, 
            unconstrained_sample
        )
    )
end

# state initialization
# note: this occurs *after* the `Pigeons.Shared` object is created, and therefore
# after `Pigeons.create_path` is called. Hence, `prototype_kernel_state` is
# already populated
# note: we initialize the RNG key using the replica's rng
function Pigeons.initialization(path::NumPyroPath, replica_rng, ::Integer)
    kernel_state = path.prototype_kernel_state._replace(
        rng_key=jax_rng_key(replica_rng)
    )
    _sample_iid(path, kernel_state)
end

# implement Pigeons.sample_iid! interface
function Pigeons.sample_iid!(::NumPyroLogPotential, replica, shared)
    path = shared.tempering.path
    kernel_state = replica.state.kernel_state
    replica.state = _sample_iid(path, kernel_state)
    return
end

function Pigeons.step!(explorer::NumPyroExplorer, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    path = shared.tempering.path

    # call the local kernel's `sample` method `n_refresh` times
    new_kernel_state = bridge.loop_sample(
        log_potential.local_kernel,
        replica.state.kernel_state,
        explorer.n_refresh,
        path.model_args,
        path.model_kwargs
    )

    # update the replica state
    replica.state = NumPyroState(new_kernel_state)

    # maybe record the sample
    if haskey(replica.recorders, :numpyro_trace) && 
        Pigeons.is_target(shared.tempering.swap_graphs, replica.chain)
        
        record_sample!(
            path, 
            replica.recorders[:numpyro_trace], 
            new_kernel_state, 
            shared.iterators.scan
        )
    end
    
    # TODO: record adaptation stuff
    return    
end

# Note: this does not work because JAX will trace the same function from all
# threads at the same time
# # Handle multithreading safely to avoid deadlocks; see
# # https://juliapy.github.io/PythonCall.jl/dev/pythoncall/#jl-multi-threading
# Pigeons.explore!(pt, explorer::NumPyroExplorer, ::Val{true}) =
#     PythonCall.GIL.@unlock Threads.@threads for replica in Pigeons.locals(pt.replicas)
#         PythonCall.GIL.@lock Pigeons.explore!(pt, replica, explorer)
#     end



# TODO: explorer adaptation
# function Pigeons.adapt_explorer(
#     explorer::NumPyroExplorer, 
#     reduced_recorders, 
#     current_pt, 
#     new_tempering
#     )
# end
