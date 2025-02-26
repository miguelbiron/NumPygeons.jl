"""
$SIGNATURES

Provides a linear prior-posterior path for NumPyro models.

$FIELDS
"""
struct NumPyroPath
    """
    A NumPyro MCMC kernel linked to the model of interest. This follows the 
    NumPyro convention that the model should be passed to this kernel's 
    constructor.
    """
    kernel::Py

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
    An example of a kernel state that allows us to initialize other kernels
    targeting tempered versions of a NumPyro model in `Pigeons.interpolate`.
    """
    prototype_kernel_state::Py

    """
    A pre-seeded python function that when called produces an unconstrained
    sample from the prior
    """
    prior_sampler::Py

    """
    A function for transforming unconstrained samples into constrained space,
    potentially adding other deterministic quantities.
    """
    sample_extractor::Py
end

"""
$SIGNATURES

Create a [`NumPyroPath`](@ref) from model arguments.
"""
function NumPyroPath(
    kernel::Py, 
    model_args::Py = pytuple(()), 
    model_kwargs::Py = pydict()
    )
    @assert is_python_tuple(model_args) "`model_args` should be a python tuple."
    @assert is_python_dict(model_kwargs) "`model_args` should be a python dict."
    
    # put placeholders in the rest of the fields; resolve in `create_path`
    NumPyroPath(
        kernel, model_args, model_kwargs, PythonCall.pynew(), 
        PythonCall.pynew(), PythonCall.pynew(), PythonCall.pynew()
    )
end

function Pigeons.create_path(path::NumPyroPath, inp::Inputs)
    @assert !inp.multithreaded """
    Multithreading is not supported (race conditions occur during JAX tracing)
    """

    # check we have a valid NumPyro MCMC kernel
    kernel = path.kernel
    @assert kernel isa Py && pyisinstance(kernel, numpyro.infer.mcmc.MCMCKernel)

    # make interpolator function
    interpolator = bridge.make_interpolator(
        kernel.model, path.model_args, path.model_kwargs
    )

    # make the sample extractor function
    sample_extractor = bridge.make_sample_extractor(
        kernel.model, 
        path.model_args, 
        path.model_kwargs,
    )

    # build the prior sampler
    rng_keys = jax.random.split(jax_rng_key(SplittableRandom(inp.seed)))
    prior_sampler = bridge.make_prior_sampler(
        kernel.model, 
        path.model_args, 
        path.model_kwargs,
        rng_keys[0]
    )

    # init the kernel
    prototype_kernel_state = path.kernel.init(
        rng_keys[1], 
        pyint(0), 
        pybuiltins.None, 
        path.model_args, 
        path.model_kwargs
    )

    # update path fields and return
    PythonCall.pycopy!(path.interpolator, interpolator)
    PythonCall.pycopy!(path.prototype_kernel_state, prototype_kernel_state)
    PythonCall.pycopy!(path.prior_sampler, prior_sampler)
    PythonCall.pycopy!(path.sample_extractor, sample_extractor)
    return path
end

Pigeons.default_explorer(::NumPyroPath) = NumPyroExplorer()
