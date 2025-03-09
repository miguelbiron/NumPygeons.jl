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

    """
    The slot in the kernel's state that points to the model's latent values.
    """
    sample_field::Py
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
        model,model_args, model_kwargs, kernel_type, kernel_kwargs,
        PythonCall.pynew(), PythonCall.pynew(), PythonCall.pynew(), 
        PythonCall.pynew(), PythonCall.pynew()
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
    # ensure pigeons was called with valid inputs
    check_inputs(inp)

    # make interpolator function
    interpolator = bridge.make_interpolator(
        path.model, path.model_args, path.model_kwargs
    )

    # make the sample extractor function
    sample_extractor = bridge.make_sample_extractor(
        path.model, path.model_args, path.model_kwargs,
    )

    # make two JAX rng keys from the seed in the inputs
    # uses a Pigeons utility function that expects a SplittableRandom
    rng_keys = jax.random.split(jax_rng_key(SplittableRandom(inp.seed)))

    # build the prior sampler, seeding it
    prior_sampler = bridge.make_prior_sampler(
        path.model, path.model_args, path.model_kwargs, rng_keys[0]
    )

    # build a temp kernel and initialize it to get a prototype_kernel_state
    temp_kernel = bridge.make_kernel_from_model(
        path.model, path.kernel_type, path.kernel_kwargs
    )
    sample_field = temp_kernel.sample_field
    prototype_kernel_state = temp_kernel.init(
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
    PythonCall.pycopy!(path.sample_field, sample_field)
    return path
end

Pigeons.default_explorer(::NumPyroPath) = NumPyroExplorer()

# Not needed, moved to dill
# # PythonCall uses `pickle` to serialize Py objects. The problem with this is
# # that closures are not handled nicely by pickle 
# # (see https://stackoverflow.com/q/72988091/5443023)
# # For NumPyroPath, this means that `prior_sampler` and `sample_extractor`
# # fail under the default serialization. Hence, we need to write specialized
# # methods to deal with them
# function Serialization.serialize(s::AbstractSerializer, path::NumPyroPath)
#     # serialize type
#     Serialization.writetag(s.io, Serialization.OBJECT_TAG)
#     Serialization.serialize(s, typeof(path))

#     # serialize contents
#     Serialization.serialize(s, path.model)
#     Serialization.serialize(s, path.model_args)
#     Serialization.serialize(s, path.model_kwargs)
#     Serialization.serialize(s, path.kernel_type)
#     Serialization.serialize(s, path.kernel_kwargs)
#     Serialization.serialize(s, path.interpolator)
#     Serialization.serialize(s, path.prototype_kernel_state)
    
#     # to handle the sampler, we just store the value of the internal rng_key
#     # we can then rebuild it during deserialization
#     rng_key = path.prior_sampler.__closure__[2].cell_contents.fn.rng_key
#     Serialization.serialize(s, rng_key)

#     # bypass the sample extractor since we recreate it during deserialization 

#     Serialization.serialize(s, path.sample_field)
# end

# function Serialization.deserialize(s::AbstractSerializer, ::Type{<:NumPyroPath})
#     model = Serialization.deserialize(s)
#     model_args = Serialization.deserialize(s)
#     model_kwargs = Serialization.deserialize(s)
#     kernel_type = Serialization.deserialize(s)
#     kernel_kwargs = Serialization.deserialize(s)
#     interpolator = Serialization.deserialize(s)
#     prototype_kernel_state = Serialization.deserialize(s)
    
#     # reconstruct the prior sampler and sample extractor
#     rng_key= Serialization.deserialize(s)
#     prior_sampler = bridge.make_prior_sampler(
#         model, model_args, model_kwargs, rng_key
#     )
#     sample_extractor = bridge.make_sample_extractor(
#         model, model_args, model_kwargs
#     )

#     sample_field = Serialization.deserialize(s)
#     return NumPyroPath(
#         model, model_args, model_kwargs, kernel_type, kernel_kwargs, 
#         interpolator, prototype_kernel_state, prior_sampler, sample_extractor,
#         sample_field
#     )
# end
