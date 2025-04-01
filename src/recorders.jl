###############################################################################
# recorders methods
###############################################################################

#######################################
# kernel adaptation
#######################################

"""
$SIGNATURES

Implements Pigeons recorder interface to provide round-based adaptation for 
compatible `MCMCKernel`s. Currently, the only supported kernels are the ones
from the [autostep](https://github.com/UBC-Stat-ML/autostep) package. 

$FIELDS
"""
mutable struct NumPyroAdaptStats
    """
    A [pytree](https://docs.jax.dev/en/latest/pytrees.html)-compatible python
    object tracking adaptation metrics.
    """
    adapt_stats::Py
end
NumPyroAdaptStats() = NumPyroAdaptStats(autostep.statistics.AutoStepAdaptStats())

"""
$SIGNATURES

Recorder builder for [`NumPyroAdaptStats`](@ref).
"""
numpyro_adapt_stats() = NumPyroAdaptStats()

# since the recorder builder is necessarily 0-argument, we need to deal with
# the initialization of the recorder
is_initialized(nas::NumPyroAdaptStats) = 
    !PythonCall.Core.pyisnone(nas.adapt_stats.means_flat)

# initialization. This should happen only once per run, since the `empty!` method
# can reuse the shape information
function initialize!(nas::NumPyroAdaptStats, kernel_state::Py)
    nas.adapt_stats = autostep.statistics.empty_adapt_stats_recorder(
        kernel_state.stats.adapt_stats
    )
    return
end

# note: `merge` needs to always create a new object. Otherwise in some corner
# cases the result is deleted when the in-place `empty!` function is called.
# E.g. when only one replica was at the target in a round. 
Base.merge(nas1::NumPyroAdaptStats, nas2::NumPyroAdaptStats) =
    return if !is_initialized(nas1)
        NumPyroAdaptStats(nas2.adapt_stats)
    elseif !is_initialized(nas2)
        NumPyroAdaptStats(nas1.adapt_stats)
    else
        NumPyroAdaptStats(bridge.merge_adapt_stats(
            nas1.adapt_stats, nas2.adapt_stats
        ))
    end

function Base.empty!(nas::NumPyroAdaptStats)
    # emptying an unitialized recorder silently failes (creates singleton 
    # arrays). no need to empty them since they already are.
    if is_initialized(nas)
        nas.adapt_stats = autostep.statistics.empty_adapt_stats_recorder(
            nas.adapt_stats
        )
    end
    return
end

#######################################
# traces
#######################################

"""
$SIGNATURES

Implements Pigeons recorder interface to provide traces for NumPyro models.

$FIELDS
"""
struct NumPyroTrace
    """
    A list of samples in constrained space.
    """
    samples_list::Py
end

NumPyroTrace() = NumPyroTrace(pylist())

"""
$SIGNATURES

Recorder builder for [`NumPyroTrace`](@ref).
"""
numpyro_trace() = NumPyroTrace()

function record_sample!(path, trace_recorder, unconstrained_sample, scan_idx)
    constrained_sample = bridge.sample_extractor(
        path.model,
        path.model_args, 
        path.model_kwargs, 
        unconstrained_sample, 
        jax.numpy.array(pyint(scan_idx))
    )
    trace_recorder.samples_list.append(constrained_sample)
    return
end

Base.merge(trace_1::NumPyroTrace, trace_2::NumPyroTrace) =
    NumPyroTrace(trace_1.samples_list + trace_2.samples_list)

Base.empty!(trace::NumPyroTrace) = trace.samples_list.clear()

"""
$SIGNATURES

Extract samples from the target defined by a NumPyro model. The output mimics
the one provided by `numpyro.infer.mcmc.get_samples`, meaning that a python
dictionary is returned, with entries given by the latent variables of the model.
Assumes that a [`NumPyroTrace`](@ref) recorder was included.
"""
function Pigeons.get_sample(
    pt::PT{<:Inputs{NumPyroPath}},
    chain = first(Pigeons.target_chains(pt))
    )
    if chain != first(Pigeons.target_chains(pt))
        error("Storage of non-target chain samples is not yet implemented")
    end
    if !haskey(pt.reduced_recorders, :numpyro_trace)
        error("""
        No NumPyroTrace found in recorders. Add the `numpyro_trace` 
        recorder builder to `pigeons(..., record=[...])`.
        """)
    end
    return bridge.stack_samples(
        pt.reduced_recorders[:numpyro_trace].samples_list
    )
end
