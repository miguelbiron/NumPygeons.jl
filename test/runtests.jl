include("setup.jl")

# toy unid example
model, model_args, model_kwargs = pyimport("numpygeons.toy_examples").toy_unid_example()
model, model_args, model_kwargs = pyimport("numpygeons.toy_examples").toy_unid_example()
kernel_type = pyimport("autostep.autohmc").AutoMALA
kernel_kwargs = pydict(;selector=pyimport("autostep.selectors").AsymmetricSelector())
path = NumPyroPath(;model,model_args, model_kwargs,kernel_type,kernel_kwargs)
true_logZ = unid_target_exact_logZ(100,50)

@testset "initialization: params and rng keys differ across replicas" begin
    pt = pigeons(target = path, n_chains = 4, n_rounds=0)
    @test allunique(r.state.kernel_state.rng_key for r in pt.replicas)
    @test allunique(r.state.kernel_state.x for r in pt.replicas)
end

# actual run
pt = pigeons(
    target = path, 
    n_chains = 4, 
    record=[numpyro_trace;record_default()]
)

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
    @test isapprox(Pigeons.stepping_stone(pt), true_logZ, rtol=0.05)
end

@testset "traces are captured" begin
    samples = get_sample(pt)
    @test jax_allclose(
        samples["__scan__"],
        NumPygeons.jax.numpy.arange(1,1+2^pt.inputs.n_rounds),
    )
    @test isapprox(jax_singleton_to_jl_float(samples["p1"].mean()), 0.7, rtol=0.05)
    @test isapprox(jax_singleton_to_jl_float(samples["p2"].mean()), 0.7, rtol=0.05)
    @testset "Deterministic values are captured" begin
        @test isapprox(jax_singleton_to_jl_float(samples["p"].mean()), 0.5, rtol=0.05)
    end
end

@testset "ChildProcess run" begin
    result = pigeons(
        target = path, 
        n_chains = 4,
        n_rounds = 2,
        checkpoint = true,
        record=[numpyro_trace;record_default()],
        on = ChildProcess(
            n_threads=1,
            dependencies=[PythonCall,NumPygeons],
            n_local_mpi_processes=2
            )
        )
    pt = Pigeons.load(result)
    @test NumPygeons.is_python_dict(get_sample(pt))
end
