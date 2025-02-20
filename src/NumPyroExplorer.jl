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

# TODO: explorer interface
# Design
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
function Pigeons.step!(explorer::NumPyroExplorer, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    state = replica.state
    kernel = explorer.kernel

    # TODO: this would only run once but we want to have multiple steps
    # write python code to achieve that (most likely using JAX scan), then call 
    # from here
    kernel._potential_fn = log_potential.potential
    new_state = kernel.sample(state, model_args, model_kwargs)
end
# function Pigeons.adapt_explorer(
#     explorer::NumPyroExplorer, 
#     reduced_recorders, 
#     current_pt, 
#     new_tempering
#     )
# end