###############################################################################
# type declarations and constructors
###############################################################################

"""
$SIGNATURES

Defines a tempered version of a NumPyro model. Note: given the tight 
association in NumPyro between kernels and target potential functions, we use 
the Pigeons log_potential interface to carry local kernels that sample from 
tempered potentials, with a local kernel state as well for adaptation.

$FIELDS
"""
mutable struct NumPyroLogPotential{T<:Real}
    """
    A local `MCMCKernel` that targets a tempered version of the original model.
    """
    local_kernel::Py

    """
    A local kernel state. It should only be used to keep track of adaptation
    stats pertaining to a fixed temperature. Do not use its rng key or sample
    field!
    """
    local_kernel_state::Py
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

A Pigeons explorer for NumPyro targets.

$FIELDS
"""
struct NumPyroExplorer
    """
    Number of times that the `sample` method of the kernel is invoked per
    exploration step.
    """
    n_refresh::Py
end

NumPyroExplorer(;n_refresh::Int = 3) = NumPyroExplorer(pyint(n_refresh))
Pigeons.explorer_recorder_builders(::NumPyroExplorer) = [numpyro_adapt_stats]

###############################################################################
# log potential methods
###############################################################################

# Path interpolation
function Pigeons.interpolate(path::NumPyroPath, beta::Real)
    potential_fn = path.interpolator(pyfloat(beta))
    local_kernel = bridge.make_kernel_from_potential(
        potential_fn, path.kernel_type, path.kernel_kwargs
    )
    init_params = pygetattr(
        path.prototype_kernel_state, 
        path.sample_field
    )

    # init local kernel and local kernel state
    # Note: the purpose of the local state is to hold adaptation information
    # DO NOT USE THE `sample_field` NOR `rng_key` IN THIS LOCAL KERNEL STATE
    # Recall that in NumPyro these fields live inside the kernel state
    # But since we must adapt to a fixed replica.chain, we need to be able to 
    # keep track of stats independent of replica.state.kernel_state
    local_kernel_state = local_kernel.init(
        # NB: since this field is not supposed to be used, not splitting the key is
        # intentional, so that any downstream reuse will be flagged as error
        path.prototype_kernel_state.rng_key,
        pyint(0), 
        init_params, 
        pybuiltins.None,
        pybuiltins.None
    )

    return NumPyroLogPotential{typeof(beta)}(local_kernel, local_kernel_state)
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
    # note: this function is initialized with its own RNG; see `create_path`
    unconstrained_sample = path.prior_sampler()
    return NumPyroState(
        bridge.update_sample_field(
            kernel_state, 
            path.sample_field, 
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
# note: since no MCMC simulation takes place here, we don't need to worry
# about recording sample statistics and can therefore use the replica state
# as is.
function Pigeons.sample_iid!(::NumPyroLogPotential, replica, shared)
    path = shared.tempering.path
    kernel_state = replica.state.kernel_state
    replica.state = _sample_iid(path, kernel_state)
    return
end

function Pigeons.step!(explorer::NumPyroExplorer, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    path = shared.tempering.path

    # make new kernel state by updating the local kernel state with the 
    # replica's sample field and rng_key 
    kernel_state = bridge.merge_kernel_states(
        log_potential.local_kernel_state, 
        replica.state.kernel_state,
        path.sample_field
    )

    # call the local kernel's `sample` method `n_refresh` times
    new_kernel_state = bridge.loop_sample(
        log_potential.local_kernel,
        kernel_state,
        explorer.n_refresh,
        path.model_args,
        path.model_kwargs
    )

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

    # at end of round, record the stats of the new_kernel_state so that
    # they can be used to create adapted kernel state in the next round
    if shared.iterators.scan == Pigeons.n_scans_in_round(shared.iterators)
        Pigeons.@record_if_requested!(
            replica.recorders, 
            :numpyro_adapt_stats,
            (replica.chain, new_kernel_state.stats.adapt_stats) 
        )
    end

    # update the log_potential's and replica's kernel_state state and return
    log_potential.local_kernel_state = new_kernel_state
    replica.state = NumPyroState(new_kernel_state)
    return
end

adapt_kernel_params(kernel, kernel_state) =
    return if pycallable(pygetattr(kernel, "adapt", pybuiltins.None))
        # note this only works for kernels in autostep package
        kernel.adapt(kernel_state, pybool(true))
    else
        kernel_state
    end

# Note: this does not work because JAX will trace the same function from all
# threads at the same time
# # Handle multithreading safely to avoid deadlocks; see
# # https://juliapy.github.io/PythonCall.jl/dev/pythoncall/#jl-multi-threading
# Pigeons.explore!(pt, explorer::NumPyroExplorer, ::Val{true}) =
#     PythonCall.GIL.@unlock Threads.@threads for replica in Pigeons.locals(pt.replicas)
#         PythonCall.GIL.@lock Pigeons.explore!(pt, replica, explorer)
#     end

