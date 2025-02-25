using Test

using NumPygeons
using Pigeons

base_dir = dirname(dirname(pathof(NumPygeons)))

# TODO: need to build python venv in general 
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(base_dir, "numpygeons", ".venv/bin/python")

using PythonCall

# analytic log normalization function for the toy unid example
include(
    joinpath(
        base_dir, 
        "test", 
        "supporting", 
        "analytic_solutions.jl"
    )
)
