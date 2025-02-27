# TODO: need to build python venv in general 
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(
    dirname(@__DIR__), "numpygeons", ".venv", "bin", "python"
)

using Test
using NumPygeons
using Pigeons
using PythonCall

# analytic log normalization function for the toy unid example
include(
    joinpath(
        dirname(dirname(pathof(Pigeons))), 
        "test", 
        "supporting", 
        "analytic_solutions.jl"
    )
)

# utils
jax_allclose(a::Py,b::Py;rtol=1e-05, atol=1e-08) = 
    Bool(pybool(NumPygeons.jax.numpy.allclose(a,b,rtol,atol)))

jax_singleton_to_jl_float(x::Py) = pyconvert(Float64, pyfloat(x))
