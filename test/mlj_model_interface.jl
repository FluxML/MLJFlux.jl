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
    alpha = rand(rng, Float32)
    lambda = rand(rng, Float32)
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
    y = Float32.(y)
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

    X = Tables.table(rand(Float32, 75, 2))
    y = rand(Float32, 75)
    @test_logs(
        (:error, MLJFlux.ERR_BUILDER),
        @test_throws UndefVarError(:Chains) MLJBase.fit(model, 0, X, y)
    )
end


@testset "layer does not exist for continuous input and transform does nothing" begin
    models = [
        MLJFlux.NeuralNetworkBinaryClassifier,
        MLJFlux.NeuralNetworkClassifier,
        MLJFlux.NeuralNetworkRegressor,
        MLJFlux.MultitargetNeuralNetworkRegressor,
    ]
    # table case
    X1 = (
        Column1 = Float32[1.0, 2.0, 3.0, 4.0, 5.0],
        Column4 = Float32[1.0, 2.0, 3.0, 4.0, 5.0],
        Column5 = randn(Float32, 5),
    )
    # matrix case
    X2 = rand(Float32, 5, 5)
    Xs = [X1, X2]

    y = categorical([0, 1, 0, 1, 1])
    yreg = Float32[0.1, -0.3, 0.2, 0.8, 0.9]
    ys = [y, y, yreg, yreg]
    for j in eachindex(Xs)
        for i in eachindex(models)
            clf = models[1](
                builder = MLJFlux.Short(n_hidden = 5, dropout = 0.2, σ = relu),
                optimiser = Optimisers.Adam(0.01),
                batch_size = 8,
                epochs = 100,
                acceleration = CUDALibs(),
                optimiser_changes_trigger_retraining = true,
            )

            mach = machine(clf, Xs[j], ys[1])

            fit!(mach, verbosity = 0)

            @test typeof(fitted_params(mach).chain.layers[1][1]) ==
                  typeof(Dense(3 => 5, relu))

            @test transform(mach, Xs[j]) == Xs[j]
        end
    end
end

@testset "transform works properly" begin
    # In this test we assumed that get_embedding_weights works
    # properly which has been tested.
    models = [
        MLJFlux.NeuralNetworkBinaryClassifier,
        MLJFlux.NeuralNetworkClassifier,
        MLJFlux.NeuralNetworkRegressor,
        MLJFlux.MultitargetNeuralNetworkRegressor,
    ]

    X = (
        Column1 = Float32[1.0, 2.0, 3.0, 4.0, 5.0],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = Float32[1.0, 2.0, 3.0, 4.0, 5.0],
        Column4 = randn(Float32, 5),
        Column5 = categorical(["group1", "group1", "group2", "group2", "group3"]),
    )

    y = categorical([0, 1, 0, 1, 1])
    yreg = Float32[0.1, -0.3, 0.2, 0.8, 0.9]
    ys = [y, y, yreg, yreg]

    for i in eachindex(models)
        clf = models[1](
            builder = MLJFlux.Short(n_hidden = 5, dropout = 0.2),
            optimiser = Optimisers.Adam(0.01),
            batch_size = 8,
            epochs = 100,
            acceleration = CUDALibs(),
            optimiser_changes_trigger_retraining = true,
            embedding_dims = Dict(:Column2 => 4, :Column5 => 2),
        )

        mach = machine(clf, X, ys[1])
        fit!(mach, verbosity = 0)
        Xenc = transform(mach, X)
        mat_col2 =
            hcat(
                [
                    collect(Xenc.Column2_1),
                    collect(Xenc.Column2_2),
                    collect(Xenc.Column2_3),
                    collect(Xenc.Column2_4),
                ]...,
            )'
        mat_col5 = hcat(
            [
                collect(Xenc.Column5_1),
                collect(Xenc.Column5_2),
            ]...,
        )'[:, [1, 3, 5]]

        mapping_matrices = MLJFlux.get_embedding_matrices(
            fitted_params(mach).chain,
            [2, 5],
            [:Column1, :Column2, :Column3, :Column4, :Column5],
        )
        mat_col2_golden = mapping_matrices[:Column2]
        mat_col5_golden = mapping_matrices[:Column5]
        @test mat_col2 == mat_col2_golden
        @test mat_col5 == mat_col5_golden
    end
end

@testset "fit, refit and predict work tests" begin
    models = [
        MLJFlux.NeuralNetworkBinaryClassifier,
        MLJFlux.NeuralNetworkClassifier,
        MLJFlux.NeuralNetworkRegressor,
        MLJFlux.MultitargetNeuralNetworkRegressor,
    ]

    X = (
        Column1 = Float32[1.0, 2.0, 3.0, 4.0, 5.0],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = Float32[1.0, 2.0, 3.0, 4.0, 5.0],
        Column4 = randn(Float32, 5),
        Column5 = categorical(["group1", "group1", "group2", "group2", "group3"]),
    )

    y = categorical([0, 1, 0, 1, 1])
    yreg = Float32[0.1, -0.3, 0.2, 0.8, 0.9]
    ys = [y, y, yreg, yreg]

    for i in eachindex(models)
        clf = models[1](
            builder = MLJFlux.Short(n_hidden = 5, dropout = 0.2),
            optimiser = Optimisers.Adam(0.01),
            batch_size = 8,
            epochs = 2,
            acceleration = CUDALibs(),
            optimiser_changes_trigger_retraining = true,
            embedding_dims = Dict(:Column2 => 4, :Column5 => 2),
        )

        mach = machine(clf, X, ys[1])
        @test_throws MLJBase.NotTrainedError mapping_matrices =
            MLJFlux.get_embedding_matrices(
                fitted_params(mach).chain,
                [2, 5],
                [:Column1, :Column2, :Column3, :Column4, :Column5],
            )
        fit!(mach, verbosity = 0)
        mapping_matrices_fit = MLJFlux.get_embedding_matrices(
            fitted_params(mach).chain,
            [2, 5],
            [:Column1, :Column2, :Column3, :Column4, :Column5],
        )
        clf.epochs = clf.epochs + 3
        clf.optimiser = Optimisers.Adam(clf.optimiser.eta / 2)
        fit!(mach, verbosity = 0)
        mapping_matrices_double_fit = MLJFlux.get_embedding_matrices(
            fitted_params(mach).chain,
            [2, 5],
            [:Column1, :Column2, :Column3, :Column4, :Column5],
        )
        @test mapping_matrices_fit != mapping_matrices_double_fit
        # Try model prediction
        Xpred = predict(mach, X)
    end
end

true
