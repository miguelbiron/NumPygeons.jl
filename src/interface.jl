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
    An MCMC kernel that is local to a replica.
    """
    local_kernel::Py

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
function Pigeons.interpolate(::NumPyroPath, beta::Real)
    return NumPyroLogPotential{typeof(beta)}(jax.numpy.array(pyfloat(beta)))
end

# log_potential evaluation
function (log_potential::NumPyroLogPotential{T})(state::NumPyroState) where T
    local_kernel = state.local_kernel
    kernel_state = state.kernel_state
    unconstrained_sample = pygetattr(kernel_state, local_kernel.sample_field)
    pot_val = local_kernel.tempered_potential(
        unconstrained_sample, log_potential.inv_temp
    )
    # convert singleton JAX array to a python float, then to Julia float
    return -pyconvert(T, pyfloat(pot_val))
end

###############################################################################
# sampling methods
###############################################################################

# iid sampling from the prior, for initialization and refreshment steps
function _sample_iid(path::NumPyroPath, local_kernel, kernel_state)
    new_kernel_state = bridge.sample_iid_kernel_state(
        path.model, 
        path.model_args, 
        path.model_kwargs, 
        local_kernel.sample_field, 
        kernel_state
    )
    return NumPyroState(local_kernel, new_kernel_state)
end

# state initialization
# note: this occurs *after* the `Pigeons.Shared` object is created, and therefore
# after `Pigeons.create_path` is called.
function Pigeons.initialization(path::NumPyroPath, replica_rng, ::Integer)
    local_kernel = bridge.make_kernel_from_model(
        path.model, path.kernel_type, path.kernel_kwargs
    )
    kernel_state = local_kernel.init(
        jax_rng_key(replica_rng),
        pyint(0), 
        pybuiltins.None, 
        path.model_args, 
        path.model_kwargs
    )
    return _sample_iid(path, local_kernel, kernel_state)
end

# implement Pigeons.sample_iid! interface
function Pigeons.sample_iid!(::NumPyroLogPotential, replica, shared)
    path = shared.tempering.path
    local_kernel = replica.state.local_kernel
    kernel_state = replica.state.kernel_state
    replica.state = _sample_iid(path, local_kernel, kernel_state)
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

    # call the local kernel's `sample` method `n_refresh` times
    local_kernel = replica.state.local_kernel
    path = shared.tempering.path
    new_kernel_state = bridge.loop_sample(
        local_kernel,
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
                pygetattr(new_kernel_state, local_kernel.sample_field), 
                shared.iterators.scan
            )
        end
    end

    # update the replica state and return
    replica.state = NumPyroState(local_kernel, new_kernel_state)
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
    as = reduced_recorders[:numpyro_adapt_stats].adapt_stats
    for replica in Pigeons.locals(pt.replicas)
        # put the adaptation statistics in the replica's kernel state
        kernel_state_with_as, _ = bridge.swap_adapt_stats(
            replica.state.kernel_state, as
        )

        # run the local kernel adaptation routine and store the new state
        # in the replica
        local_kernel = replica.state.local_kernel
        adapted_kernel_state = local_kernel.adapt(kernel_state_with_as, true)
        replica.state = NumPyroState(local_kernel, adapted_kernel_state)
    end
    return explorer
end
