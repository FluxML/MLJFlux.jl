Random.seed!(123)

N = 200
X = MLJBase.table(randn(Float32, N, 5));

builder = MLJFlux.Short(σ=identity)
optimiser = Optimisers.Adam()

Random.seed!(123)
y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5
train, test = MLJBase.partition(1:N, 0.7)

@testset_accelerated "NeuralNetworkRegressor" accel begin

    # Table input:
    @testset "Table input" begin
        basictest(
            MLJFlux.NeuralNetworkRegressor,
            X,
            y,
            builder,
            optimiser,
            0.7,
            accel,
        )
    end

    # Matrix input:
    @testset "Matrix input" begin
        @test basictest(
            MLJFlux.NeuralNetworkRegressor,
            matrix(X),
            y,
            builder,
            optimiser,
            0.7,
            accel,
        )
    end

    # test model is a bit better than constant predictor:
    # (GPUs only support `default_rng` when there's `Dropout`):
    rng = Random.default_rng()
    seed!(rng, 123)
    model = MLJFlux.NeuralNetworkRegressor(builder=builder,
                                           acceleration=accel,
                                           rng=rng)
    @time fitresult, _, rpt =
        fit(model, 0, MLJBase.selectrows(X, train), y[train])
    first_last_training_loss = rpt[1][[1, end]]
#    @show first_last_training_loss
    yhat = predict(model, fitresult, selectrows(X, test))
    truth = y[test]
    goal = 0.9*model.loss(truth .- mean(truth), 0)
    @test model.loss(yhat, truth) < goal
end

Random.seed!(123)
ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5);
y = MLJBase.table(ymatrix);

@testset_accelerated "MultitargetNeuralNetworkRegressor" accel begin

    # Table input:
    @testset "Table input" begin
        @test basictest(
            MLJFlux.MultitargetNeuralNetworkRegressor,
            X,
            y,
            builder,
            optimiser,
            1.0,
            accel,
        )
    end
    # Matrix input:
    @testset "Matrix input" begin
        @test basictest(
            MLJFlux.MultitargetNeuralNetworkRegressor,
            matrix(X),
            ymatrix,
            builder,
            optimiser,
            1.0,
            accel,
        )
    end

    # test model is a bit better than constant predictor
    # (GPUs only support `default_rng` when there's `Dropout`):
    rng = Random.default_rng()
    seed!(rng, 123)
    model = MLJFlux.MultitargetNeuralNetworkRegressor(
        acceleration=accel,
        builder=builder,
        rng=rng,
    )
    @time fitresult, _, rpt =
        fit(model, 0, MLJBase.selectrows(X, train), selectrows(y, train))
    first_last_training_loss = rpt[1][[1, end]]
    yhat = predict(model, fitresult, selectrows(X, test))
    truth = ymatrix[test,:]
    goal = 0.85*model.loss(truth .- mean(truth), 0)
    @test model.loss(Tables.matrix(yhat), truth) < goal
end

true
