# input validation
is_python_tuple(t) = (t isa Py && pyisinstance(t, pytype(pytuple(()))))
is_python_dict(d) = (d isa Py && pyisinstance(d, pytype(pydict())))

"""
$SIGNATURES

Make a JAX random number generator (RNG) key from a Julia RNG.
"""
jax_rng_key(rng) = jax.random.key(pyint(Pigeons.java_seed(rng)))
