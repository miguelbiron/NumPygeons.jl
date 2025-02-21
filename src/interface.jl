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
    @assert model_args isa Py && pyisinstance(model_args, pytype(pytuple(()))),
        "`model_args` should be a python tuple."
    @assert model_kwargs isa Py && pyisinstance(model_kwargs, pytype(pydict())),
        "`model_args` should be a python dict."
    
    # put placeholders in the rest of the fields; resolve in `create_path`
    NumPyroPath(
        model_args, model_kwargs, PythonCall.pynew(), PythonCall.pynew()
    )
end

# Update the fields in the path using the explorer
function Pigeons.create_path(
    path::NumPyroPath,
    inp::Inputs{NumPyroPath,<:Any,NumPyroExplorer}
    )
    # check we have a valid NumPyro MCMC kernel
    # note: kernel is later initialized in `Pigeons.initialization`
    kernel = inp.explorer.kernel
    @assert kernel isa Py && pyisinstance(kernel, numpyro.infer.MCMC)

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
    return initial_kernel_state
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

Pigeons.interpolate(path::NumPyroPath, beta::Real) = 
    NumPyroLogPotential(path.kernel, path.interpolator(pyfloat(beta)))

function (log_potential::NumPyroLogPotential)(state::NumPyroState)
    kernel = log_potential.kernel
    potential_fn = log_potential.potential_fn
    kernel_state = state.kernel_state
    unconstrained_sample = pygetattr(kernel_state, kernel.sample_field)
    return -pyconvert(Float64, potential_fn(unconstrained_sample))
end
