Random.seed!(123)

N = 200
X = MLJBase.table(randn(Float32, N, 5));

# TODO: replace Short2 -> Short when
# https://github.com/FluxML/Flux.jl/pull/1618 is resolved:
builder = Short2(σ=identity)
optimiser = Flux.Optimise.ADAM()

losses = []

Random.seed!(123)
y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5
train, test = MLJBase.partition(1:N, 0.7)

@testset_accelerated "NeuralNetworkRegressor" accel begin

    Random.seed!(123)

    basictest(MLJFlux.NeuralNetworkRegressor,
              X,
              y,
              builder,
              optimiser,
              0.7,
              accel)

    # test model is a bit better than constant predictor:
    stable_rng = StableRNGs.StableRNG(123)
    model = MLJFlux.NeuralNetworkRegressor(builder=builder,
                                           acceleration=accel,
                                           rng=stable_rng)
    @time fitresult, _, rpt =
        fit(model, 0, MLJBase.selectrows(X, train), y[train])
    first_last_training_loss = rpt[1][[1, end]]
    push!(losses, first_last_training_loss[2])
#    @show first_last_training_loss
    yhat = predict(model, fitresult, selectrows(X, test))
    truth = y[test]
    goal = 0.9*model.loss(truth .- mean(truth), 0)
    @test model.loss(yhat, truth) < goal

    optimisertest(MLJFlux.NeuralNetworkRegressor,
                  X,
                  y,
                  builder,
                  optimiser,
                  accel)

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@test all(x->abs(x - reference)/reference < 1e-6, losses[2:end])

Random.seed!(123)
ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5);
y = MLJBase.table(ymatrix);

losses = []

@testset_accelerated "MultitargetNeuralNetworkRegressor" accel begin

    Random.seed!(123)

    basictest(MLJFlux.MultitargetNeuralNetworkRegressor,
              X,
              y,
              builder,
              optimiser,
              1.0,
              accel)

    # test model is a bit better than constant predictor
    model = MLJFlux.MultitargetNeuralNetworkRegressor(acceleration=accel,
                                                      builder=builder)
    @time fitresult, _, rpt =
        fit(model, 0, MLJBase.selectrows(X, train), selectrows(y, train))
    first_last_training_loss = rpt[1][[1, end]]
    push!(losses, first_last_training_loss[2])
#   @show first_last_training_loss
    yhat = predict(model, fitresult, selectrows(X, test))
    truth = ymatrix[test,:]
    goal = 0.8*model.loss(truth .- mean(truth), 0)
    @test model.loss(Tables.matrix(yhat), truth) < goal

    optimisertest(MLJFlux.MultitargetNeuralNetworkRegressor,
              X,
              y,
              builder,
              optimiser,
              accel)

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@test all(x->abs(x - reference)/reference < 1e-6, losses[2:end])

true
