"""
$SIGNATURES

Provides a linear prior-posterior path for NumPyro models.

$FIELDS
"""
struct NumPyroPath
    """
    An instance of (a subclass of) `numpyro.infer.mcmc.MCMCKernel` that will be
    used for exploration. Should be linked to the NumPyro model of interest. 
    """
    mcmc_kernel::Py

    """
    A python tuple with optional arguments to the NumPyro model.
    """
    model_args::Py

    """
    A python dictionary with optional keyword arguments to the NumPyro model.
    """
    model_kwargs::Py

    """
    Internal kernel state that allows us to initialize replicas' states.
    """
    prototype_kernel_state::Py
end

"""
$SIGNATURES

Create a [`NumPyroPath`](@ref) from model arguments.
"""
function NumPyroPath(;
    mcmc_kernel,
    model_args = pytuple(()), 
    model_kwargs = pydict()
    )
    # undo some automatic conversion when passing from python
    if model_args isa Tuple
        @info "`model_args` was a Julia Tuple; converting to python tuple"
        model_args = pytuple(model_args)
    end
    if model_kwargs isa PyDict
        @info "`model_kwargs` was a PythonCall.PyDict; converting to python dict"
        model_kwargs = pydict(model_kwargs)
    end

    @assert mcmc_kernel isa Py && pyisinstance(
        mcmc_kernel, numpyro.infer.mcmc.MCMCKernel
    )
    @assert is_python_tuple(model_args) """
        `model_args` should be a python tuple; got $(typeof(model_args))
    """
    @assert is_python_dict(model_kwargs) """
        `model_kwargs` should be a python dict; got $(typeof(model_kwargs))
    """
    
    # put placeholders in the rest of the fields; resolve in `create_path`
    NumPyroPath(
        mcmc_kernel, model_args, model_kwargs, PythonCall.pynew()
    )
end

function check_inputs(inp::Inputs)
    @assert !inp.multithreaded """
    Multithreading is not supported (race conditions occur during JAX tracing).
    """

    @assert !(Pigeons.traces in inp.record) """
    NumPygeons is incompatible with the `traces` recorder. Use the specialized
    `numpyro_trace` recorder instead.
    """
end

function Pigeons.create_path(path::NumPyroPath, inp::Inputs)
    check_inputs(inp) # ensure pigeons was called with valid inputs

    # create a prototype_kernel_state that will be used to init replicas
    prototype_kernel_state = path.mcmc_kernel.init(
        jax_rng_key(SplittableRandom(inp.seed)), # uses a Pigeons utility function that expects a SplittableRandom
        pyint(0), 
        pybuiltins.None, 
        path.model_args, 
        path.model_kwargs
    )
    PythonCall.pycopy!(path.prototype_kernel_state, prototype_kernel_state)
    return path
end

Pigeons.default_explorer(::NumPyroPath) = NumPyroExplorer()

