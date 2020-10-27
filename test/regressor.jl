Random.seed!(123)

N = 200
X = MLJBase.table(randn(Float32, N, 5));

# TODO: replace Short2 -> Short when
# https://github.com/FluxML/Flux.jl/issues/1372 is resolved:
builder = Short2(σ=identity)
optimiser = Flux.Optimise.ADAM()

losses = []

@testset_accelerated "NeuralNetworkRegressor" accel begin
    Random.seed!(123)
    y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5
    basictest(MLJFlux.NeuralNetworkRegressor,
              X,
              y,
              builder,
              optimiser,
              0.7,
              accel)

    # test a bit better than constant predictor
    model = MLJFlux.NeuralNetworkRegressor(acceleration=accel)
    train, test = MLJBase.partition(1:N, 0.7)
    @time mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    first_last_training_loss = MLJBase.report(mach)[1][[1, end]]
    push!(losses, first_last_training_loss[2])
    @show first_last_training_loss
    yhat = predict(mach, rows=test)
    truth = y[test]
    goal = 1.0*model.loss(truth .- mean(truth), 0)
    @test model.loss(yhat, truth) < goal
end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@test all(x->abs(x - reference)/x < 1e-6, losses[2:end])

@testset_accelerated "MultitargetNeuralNetworkRegressor" accel begin
    Random.seed!(123)
    ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5);
    y = MLJBase.table(ymatrix);
    basictest(MLJFlux.MultitargetNeuralNetworkRegressor,
              X,
              y,
              builder,
              optimiser,
              1.0,
              accel)

    # test a bit better than constant predictor
    model = MLJFlux.MultitargetNeuralNetworkRegressor(acceleration=accel)
    train, test = MLJBase.partition(1:N, 0.7)
    @time mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    first_last_training_loss = MLJBase.report(mach)[1][[1, end]]
    @show first_last_training_loss
    yhat = predict(mach, rows=test)
    truth = ymatrix[test]
    goal = 1.0*model.loss(truth .- mean(truth), 0)
    @test model.loss(Tables.matrix(yhat), truth) < goal
end

true
