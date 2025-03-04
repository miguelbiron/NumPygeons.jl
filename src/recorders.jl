###############################################################################
# recorders methods
###############################################################################

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

function record_sample!(path, trace_recorder, new_kernel_state, scan_idx)
    unconstrained_sample = pygetattr(new_kernel_state, path.sample_field)
    constrained_sample = path.sample_extractor(
        unconstrained_sample, 
        pyint(scan_idx)
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
