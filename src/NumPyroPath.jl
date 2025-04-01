"""
$SIGNATURES

Provides a linear prior-posterior path for NumPyro models.

$FIELDS
"""
struct NumPyroPath
    """
    A NumPyro model.
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
    A type of NumPyro MCMC kernel (i.e., a subclass of 
    `numpyro.infer.mcmc.MCMCKernel`) that will be used for exploration.
    """
    kernel_type::Py

    """
    Optional keyword arguments passed to the kernel's constructor.
    """
    kernel_kwargs::Py
end

"""
$SIGNATURES

Create a [`NumPyroPath`](@ref) from model arguments.
"""
function NumPyroPath(;
    model,
    model_args = pytuple(()), 
    model_kwargs = pydict(),
    kernel_type = pyimport("autostep.autohmc").AutoMALA,
    kernel_kwargs = pydict(), 
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
    if kernel_kwargs isa PyDict
        @info "`kernel_kwargs` was a PythonCall.PyDict; converting to python dict"
        kernel_kwargs = pydict(kernel_kwargs)
    end

    @assert kernel_type isa Py && pyisinstance(
        kernel_type(), numpyro.infer.mcmc.MCMCKernel)
    @assert is_python_tuple(model_args) """
        `model_args` should be a python tuple; got $(typeof(model_args))
    """
    @assert is_python_dict(model_kwargs) """
        `model_kwargs` should be a python dict; got $(typeof(model_kwargs))
    """
    @assert is_python_dict(kernel_kwargs) """
        `kernel_kwargs` should be a python dict; got $(typeof(kernel_kwargs))
    """
    
    # put placeholders in the rest of the fields; resolve in `create_path`
    NumPyroPath(
        model, model_args, model_kwargs, kernel_type, kernel_kwargs
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
    return path
end

Pigeons.default_explorer(::NumPyroPath) = NumPyroExplorer()

