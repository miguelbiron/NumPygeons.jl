"""
$SIGNATURES

A thin wrapper around a NumPyro model to provide a linear prior-posterior path.

$FIELDS
"""
struct NumPyroPath
    """
    A NumPyro model encapsulated in a `PythonCall.Py` object.
    """
    model::Py

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

    NumPyroPath(model::Py, model_args::Py, model_kwargs::Py) =
        new(
            model, model_args, model_kwargs,
            bridge.make_interpolator(model, model_args, model_kwargs)
        )
end

default_explorer(path::NumPyroPath) = AutoStepExplorer(path)

# target is already a Path
Pigeons.create_path(target::NumPyroPath, ::Inputs) = target

"""
$SIGNATURES

Defines an tempered version of a NumPyro model.

$FIELDS
"""
@Pigeons.auto struct NumPyroLogPotential
    """
    The path that generated this logpotential.
    """
    path

    """
    An inverse temperature
    """
    beta

    """
    A python object representing a tempered potential function. 
    Note: Following the NumPyro convention, a potential is the negative of
    a Pigeons logpotential.    
    """
    potential
end

Pigeons.interpolate(path::NumPyroPath, beta) = 
    NumPyroLogPotential(path, beta, path.interpolator(beta))

