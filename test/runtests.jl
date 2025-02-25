ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/home/mbiron/projects/NumPygeons.jl/numpygeons/.venv/bin/python"

using Test

using PythonCall
using NumPygeons
using Pigeons

# analytic log normalization function for the toy unid example
include(
    joinpath(
        dirname(dirname(pathof(Pigeons))), 
        "test", 
        "supporting", 
        "analytic_solutions.jl"
    )
)

# toy unid example
model, model_args, model_kwargs = pyimport("numpygeons.toy_examples").toy_unid_example()
autohmc = pyimport("autostep.autohmc")
kernel = autohmc.AutoMALA(model)
path = NumPyroPath(kernel,model_args, model_kwargs)

@testset "initialization: params and rng keys differ across replicas" begin
    pt = pigeons(target = path, n_chains = 4, n_rounds=0)
    @test allunique(r.state.kernel_state.rng_key for r in pt.replicas)
    @test allunique(r.state.kernel_state.x for r in pt.replicas)
end

pt = pigeons(target = path, n_chains = 4)

@testset "betas in tempered potential closures should match schedule" begin
    @test pt.shared.tempering.schedule.grids ==
        [
            pyconvert(
                eltype(pt.shared.tempering.schedule.grids),
                lp.local_kernel._potential_fn.__wrapped__.__closure__[0].cell_contents
            ) for lp in pt.shared.tempering.log_potentials
        ]
end

@testset "logZ approx is correct" begin
    @test isapprox(
        Pigeons.stepping_stone(pt), 
        unid_target_exact_logZ(100,50), 
        rtol=0.05
    )
end

