ModelType = MLJFlux.NeuralNetworkRegressor

@testset "equality" begin
    model = MLJFlux.ImageClassifier()
    clone = deepcopy(model)
    @test model == clone
    clone.optimiser = Optimisers.Adam(model.optimiser.eta*10)
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

    @test_throws MLJFlux.ERR_BAD_OPTIMISER NeuralNetworkClassifier(
        optimiser=Flux.Optimise.Adam(),
    )
end

@testset "regularization: logic" begin
    optimiser = Optimisers.Momentum()

    # lambda = 0:
    model = MLJFlux.NeuralNetworkRegressor(; alpha=0.3, lambda=0, optimiser)
    chain = MLJFlux.regularized_optimiser(model, 1)
    @test chain == optimiser

    # alpha = 0:
    model = MLJFlux.NeuralNetworkRegressor(; alpha=0, lambda=0.3, optimiser)
    chain = MLJFlux.regularized_optimiser(model, 1)
    @test chain isa Optimisers.OptimiserChain{
        Tuple{Optimisers.WeightDecay, Optimisers.Momentum}
    }

    # alpha = 1:
    model = MLJFlux.NeuralNetworkRegressor(; alpha=1, lambda=0.3, optimiser)
    chain = MLJFlux.regularized_optimiser(model, 1)
    @test chain isa Optimisers.OptimiserChain{
        Tuple{Optimisers.SignDecay, Optimisers.Momentum}
    }

    # general case:
    model = MLJFlux.NeuralNetworkRegressor(; alpha=0.4, lambda=0.3, optimiser)
    chain = MLJFlux.regularized_optimiser(model, 1)
    @test chain isa Optimisers.OptimiserChain{
        Tuple{Optimisers.SignDecay, Optimisers.WeightDecay, Optimisers.Momentum}
    }
end

@testset "regularization: integration" begin
    rng = StableRNG(123)
    nobservations = 12
    Xuser = rand(Float32, nobservations, 3)
    yuser = rand(Float32, nobservations)
    alpha = rand(rng)
    lambda = rand(rng)
    optimiser = Optimisers.Momentum()
    builder = MLJFlux.Linear()
    epochs = 1 # don't change this
    opts = (; alpha, lambda, optimiser, builder, epochs)

    for batch_size in [1, 2, 3]

        # (1) train using weight/sign decay, as implemented in MLJFlux:
        model = MLJFlux.NeuralNetworkRegressor(; batch_size, rng=StableRNG(123), opts...);
        mach = machine(model, Xuser, yuser);
        fit!(mach, verbosity=0);
        w1 = Optimisers.trainables(fitted_params(mach).chain)

        # (2) manually train for one epoch explicitly adding a loss penalty:
        chain = MLJFlux.build(builder, StableRNG(123), 3, 1);
        penalty = Penalizer(lambda, alpha); # defined in test_utils.jl
        X, y = MLJFlux.collate(model, Xuser, yuser, 0);
        loss = model.loss;
        n_batches = div(nobservations, batch_size)
        optimiser_state = Optimisers.setup(optimiser, chain);
        for i in 1:n_batches
            batch_loss, gs = Flux.withgradient(chain) do m
                yhat = m(X[i])
                loss(yhat, y[i]) + sum(penalty, Optimisers.trainables(m))/n_batches
            end
            ∇ = first(gs)
            optimiser_state, chain = Optimisers.update(optimiser_state, chain, ∇)
        end
        w2 = Optimisers.trainables(chain)

        # (3) compare the trained weights
        @test w1 ≈ w2
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
