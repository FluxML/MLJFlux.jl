Random.seed!(123)

N = 200
X = MLJBase.table(randn(Float32, N, 5));

builder = MLJFlux.Short(σ=identity)
optimiser = Flux.Optimise.ADAM()

@testset "NeuralNetworkRegressor" begin
    y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5
    basictest(MLJFlux.NeuralNetworkRegressor, X, y, builder, optimiser)

    # test a bit better than constant predictor
    model = MLJFlux.NeuralNetworkRegressor()
    train, test = MLJBase.partition(1:N, 0.7)
    mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    yhat = predict(mach, rows=test)
    truth = y[test]
    goal =0.8*model.loss(truth .- mean(truth), 0)
    @test model.loss(yhat, truth) < goal
end

@testset "MultitargetNeuralNetworkRegressor" begin
    ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5);
    y = MLJBase.table(ymatrix);
    basictest(MLJFlux.MultitargetNeuralNetworkRegressor,
              X,
              y,
              builder,
              optimiser)

    # test a bit better than constant predictor
    model = MLJFlux.MultitargetNeuralNetworkRegressor()
    train, test = MLJBase.partition(1:N, 0.7)
    mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    yhat = predict(mach, rows=test)
    truth = ymatrix[test]
    goal =0.8*model.loss(truth .- mean(truth), 0)
    @test model.loss(Tables.matrix(yhat), truth) < goal
end

true
