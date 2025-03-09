include("setup.jl")

# toy unid example
model, model_args, model_kwargs = pyimport("numpygeons.toy_examples").toy_unid_example()
kernel_kwargs = pydict(;selector=pyimport("autostep.selectors").AsymmetricSelector())
path = NumPyroPath(;model,model_args, model_kwargs,kernel_kwargs)
true_logZ = unid_target_exact_logZ(100,50)

# inspect initialization
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

@testset "logZ and Lambda are well approx." begin
    @test isapprox(Pigeons.stepping_stone(pt), true_logZ, rtol=0.05)
    @test isapprox(Pigeons.global_barrier(pt.shared.tempering), 1.39, rtol=0.05) # 16 round with Pigeons
end

@testset "adaptation" begin
    # exact number of samples are taken
    as = pt.reduced_recorders[:numpyro_adapt_stats].adapt_stats
    @test Pigeons.n_scans_in_round(pt.shared.iterators) == pyconvert(
        Int, pyint(as.sample_idx ÷ pt.shared.explorer.n_refresh)
    )
    # means and variances are roughly approximated
    # values are from a Pigeons+DynamicPPL 16 rounds run
    @test jax_allclose(as.means_flat, 1.16418003938146, atol = 0.2, rtol = 0.05)
    @test jax_allclose(as.vars_flat, 1.229661247389096, atol = 0.6, rtol = 0.05)

    # step size is adapted from mean step size from prev round, and all replicas
    # get this same value
    @test iszero(jax_singleton_to_jl_float(
        pt.replicas[end].state.kernel_state.base_step_size - as.mean_step_size
    ))
    @test allequal(
        r.state.kernel_state.base_step_size for r in pt.replicas
    )
end

@testset "traces are captured" begin
    samples = get_sample(pt)
    @test jax_allclose(
        samples["__scan__"],
        NumPygeons.jax.numpy.arange(1,1+2^pt.inputs.n_rounds),
    )
    @test isapprox(jax_singleton_to_jl_float(samples["p1"].mean()), 0.7, atol=0.05, rtol=0.1)
    @test isapprox(jax_singleton_to_jl_float(samples["p2"].mean()), 0.7, atol=0.05, rtol=0.1)
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

@testset "8 schools" begin
    model, model_args, model_kwargs = pyimport("numpygeons.toy_examples").eight_schools_example()
    path = NumPyroPath(;model,model_args, model_kwargs)
    pt = pigeons(
        target = path, 
        n_chains = 3,
        record = [numpyro_trace;record_default();energy_ac1;round_trip],
    )
    # "true" values are from 16 round run with Pigeons
    @test isapprox(Pigeons.stepping_stone(pt), -31.3, rtol=0.05)
    @test isapprox(Pigeons.global_barrier(pt.shared.tempering), 0.966, rtol=0.15)
    @test NumPygeons.is_python_dict(get_sample(pt))
end
