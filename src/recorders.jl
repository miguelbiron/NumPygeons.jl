###############################################################################
# recorders methods
###############################################################################

# recorder builder for keeping track of the adaptation statistics
numpyro_adapt_stats() = Dict{Int,Py}()

function Pigeons.record!(recorder::Dict{Int,Py}, tup::Tuple)
    chain_idx, value = tup
    recorder[chain_idx] = value
end

"""
$SIGNATURES

Implements Pigeons recorder interface to provide a traces for NumPyro models.

$FIELDS
"""
struct NumPyroTrace
    """
    A list of samples in constrained space.
    """
    samples_list::Py
end

NumPyroTrace() = NumPyroTrace(pylist())
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

function Pigeons.get_sample(
    pt::PT{<:Inputs{NumPyroPath}},
    chain = first(Pigeons.target_chains(pt))
    )
    if chain != first(Pigeons.target_chains(pt))
        error("Storage of non-target chain samples is not yet implemented")
    end
    return bridge.stack_samples(
        pt.reduced_recorders[:numpyro_trace].samples_list
    )
end
