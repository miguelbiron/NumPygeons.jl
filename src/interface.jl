###############################################################################
# type declarations and constructors
###############################################################################

"""
$SIGNATURES

Defines a tempered version of a NumPyro model.

$FIELDS
"""
struct NumPyroLogPotential{T<:Real}
    """
    A reference to the [NumPyroPath](@ref) mcmc kernel.
    """
    global_kernel::Py

    """
    A `jax` singleton array corresponding to an inverse temperature.
    """
    inv_temp::Py
end

"""
$SIGNATURES

Replica state for a NumPyro target.

$FIELDS
"""
struct NumPyroState
    """
    Full state of a replica's MCMC kernel.
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
Pigeons.interpolate(path::NumPyroPath, beta::Real) = 
    NumPyroLogPotential{typeof(beta)}(
        path.mcmc_kernel,
        jax.numpy.array(pyfloat(beta))
    )

# log_potential evaluation
function (log_potential::NumPyroLogPotential{T})(state::NumPyroState) where T
    global_kernel = log_potential.global_kernel
    kernel_state = state.kernel_state
    unconstrained_sample = pygetattr(kernel_state, global_kernel.sample_field)
    pot_val = global_kernel.tempered_potential(
        unconstrained_sample, log_potential.inv_temp
    )
    # convert singleton JAX array to a python float, then to Julia float
    return -pyconvert(T, pyfloat(pot_val))
end

###############################################################################
# sampling methods
###############################################################################

# iid sampling from the prior, for initialization and refreshment steps
function _sample_iid(path::NumPyroPath, kernel_state)
    new_kernel_state = bridge.sample_iid_kernel_state(
        path.mcmc_kernel.model, 
        path.model_args, 
        path.model_kwargs, 
        path.mcmc_kernel.sample_field, 
        kernel_state
    )
    return NumPyroState(new_kernel_state)
end

# state initialization
# note: this occurs *after* the `Pigeons.Shared` object is created, and therefore
# after `Pigeons.create_path` is called.
function Pigeons.initialization(path::NumPyroPath, replica_rng, ::Integer)
    kernel_state = path.prototype_kernel_state._replace(
        rng_key = jax_rng_key(replica_rng)
    )
    return _sample_iid(path, kernel_state)
end

# implement Pigeons.sample_iid! interface
function Pigeons.sample_iid!(::NumPyroLogPotential, replica, shared)
    path = shared.tempering.path
    replica.state = _sample_iid(path, replica.state.kernel_state)
    return
end

# explorer step
function Pigeons.step!(explorer::NumPyroExplorer, replica, shared)
    # update the inverse temperature in the kernel state
    log_potential = Pigeons.find_log_potential(
        replica, shared.tempering, shared
    )
    kernel_state = replica.state.kernel_state._replace(
        inv_temp = log_potential.inv_temp
    )

    # plug the adapt stats recorder if we are at the target
    at_target = Pigeons.is_target(shared.tempering.swap_graphs, replica.chain)
    if at_target && haskey(replica.recorders, :numpyro_adapt_stats)
        nas = replica.recorders[:numpyro_adapt_stats]
        is_initialized(nas) || initialize!(nas, kernel_state)
        kernel_state, old_adapt_stats = bridge.swap_adapt_stats(
            kernel_state,
            nas.adapt_stats
        )
    end

    # call the mcmc kernel's `sample` method `n_refresh` times
    path = shared.tempering.path
    new_kernel_state = bridge.loop_sample(
        path.mcmc_kernel,
        kernel_state,
        explorer.n_refresh,
        path.model_args,
        path.model_kwargs
    )

    # postprocessing exclusive to the target chain
    if at_target
        # remove the adaptation recorder and put it back in the replica
        if haskey(replica.recorders, :numpyro_adapt_stats)
            new_kernel_state, new_adapt_stats = bridge.swap_adapt_stats(
                new_kernel_state,
                old_adapt_stats
            )
            replica.recorders[:numpyro_adapt_stats].adapt_stats = new_adapt_stats
        end

        # maybe record the sample
        if haskey(replica.recorders, :numpyro_trace)
            record_sample!(
                path, 
                replica.recorders[:numpyro_trace], 
                pygetattr(new_kernel_state, path.mcmc_kernel.sample_field), 
                shared.iterators.scan
            )
        end
    end

    # update the replica state and return
    replica.state = NumPyroState(new_kernel_state)
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

###############################################################################
# adaptation methods
###############################################################################

# adapt the kernel states in the replicas using the adapt_stats gathered
# in the previous round. This update is done in place, as this function must
# only return an explorer
function Pigeons.adapt_explorer(
    explorer::NumPyroExplorer, 
    reduced_recorders, 
    pt, 
    updated_tempering
    )
    haskey(reduced_recorders, :numpyro_adapt_stats) || return explorer
    adapt_stats = reduced_recorders[:numpyro_adapt_stats].adapt_stats
    for replica in Pigeons.locals(pt.replicas)
        # put the adaptation statistics in the replica's kernel state
        kernel_state_with_stats, _ = bridge.swap_adapt_stats(
            replica.state.kernel_state, adapt_stats
        )

        # run the local kernel adaptation routine and store the new state
        # in the replica
        mcmc_kernel = updated_tempering.path.mcmc_kernel
        adapted_kernel_state = mcmc_kernel.adapt(kernel_state_with_stats, true)
        replica.state = NumPyroState(adapted_kernel_state)
    end
    return explorer
end
