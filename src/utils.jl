# input validation
is_python_tuple(t) = (t isa Py && pyisinstance(t, pytype(pytuple(()))))
is_python_dict(d) = (d isa Py && pyisinstance(d, pytype(pydict())))

"""
$SIGNATURES

Make a JAX random number generator (RNG) key from a Julia RNG.
"""
jax_rng_key(rng) = jax.random.key(pyint(Pigeons.java_seed(rng)))

ensure_bridge_exist() = try
    pyimport("numpygeons")
catch e
    e isa PyException || rethrow(e)
    @warn "Bridge package not found in python environment. Attempting install."
    install_bridge()
end

function install_bridge()
    bridge_path = joinpath(dirname(@__DIR__), "numpygeons")
    python_path = abspath(PythonCall.python_executable_path())
    try
        run(`$python_path -m pip --version`)
    catch e
        @warn """
        Cannot install python bridge because pip was not found. Please install 
        the bridge with your favorite environment manager from the following 
        location

            $bridge_path
        """
        rethrow(e)
    end
    run(`$python_path -m pip install -e $bridge_path`)
end
