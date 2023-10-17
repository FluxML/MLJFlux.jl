ModelType = MLJFlux.NeuralNetworkRegressor

@testset "equality" begin
    model = MLJFlux.ImageClassifier()
    clone = deepcopy(model)
    @test model == clone
    clone.optimiser.eta *= 10
    @test model != clone
end

@testset "clean!" begin
    model = @test_logs (:warn, r"`lambda") begin
        ModelType(lambda = -1)
    end
    @test model.lambda == 0

    model = @test_logs (:warn, r"`alpha") begin
        ModelType(alpha = -1)
    end
    @test model.alpha == 0

    model = @test_logs (:warn, r"`epochs") begin
        ModelType(epochs = -1)
    end
    @test model.epochs == 10

    model = @test_logs (:warn, r"`batch_size") begin
        ModelType(batch_size = 0)
    end
    @test model.batch_size == 1

    if MLJFlux.gpu_isdead()
        model  = @test_logs (:warn, r"`acceleration") match_mode = :any begin
            ModelType(acceleration=CUDALibs())
        end
        @test model.acceleration == CUDALibs()
    end
end

@testset "iteration api" begin
    model = MLJFlux.NeuralNetworkRegressor(epochs=10)
    @test MLJBase.supports_training_losses(model)
    @test MLJBase.iteration_parameter(model) == fieldnames(typeof(model))[4]

    # integration test:
    X, y = MLJBase.make_regression(10)
    X = Float32.(MLJBase.Tables.matrix(X)) |> MLJBase.Tables.table
    mach = MLJBase.machine(model, X, y)
    MLJBase.fit!(mach, verbosity=0)
    losses = MLJBase.training_losses(mach)
    @test losses == MLJBase.report(mach).training_losses[2:end]
    @test length(losses) == 10
end

mutable struct LisasBuilder
  n1::Int
end

@testset "builder errors and issue #237" begin
    # create a builder with an intentional flaw;
    # `Chains` is undefined - it should be `Chain`
    function MLJFlux.build(builder::LisasBuilder, rng, nin, nout)
        return Flux.Chains(
            Flux.Dense(nin, builder.n1),
            Flux.Dense(builder.n1, nout)
        )
    end

    model = NeuralNetworkRegressor(
        epochs = 2,
        batch_size = 32,
        builder = LisasBuilder(10),
    )

    X, y = @load_boston
    @test_logs(
        (:error, MLJFlux.ERR_BUILDER),
        @test_throws UndefVarError(:Chains) MLJBase.fit(model, 0, X, y)
    )
end

true
