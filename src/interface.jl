"""
$SIGNATURES

A thin wrapper around a NumPyro model to provide a linear prior-posterior path.

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
function NumPyroPath(model_args::Py, model_kwargs::Py)
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
    inp::Inputs{<:Any,<:Any,NumPyroExplorer}
    )
    kernel = inp.explorer.kernel
    @assert kernel isa Py && pyisinstance(kernel, numpyro.infer.MCMC)
    interpolator = bridge.make_interpolator(
        kernel.model, path.model_args, path.model_kwargs
    )
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
    potential::Py
end

Pigeons.interpolate(path::NumPyroPath, beta::Real) = 
    NumPyroLogPotential(path.kernel, path.interpolator(pyfloat(beta)))

function (lp::NumPyroLogPotential)(x::NumPyroState)
    # TODO
end