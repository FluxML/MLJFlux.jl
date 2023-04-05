## NEURAL NETWORK CLASSIFIER

seed!(1234)
N = 300
X = MLJBase.table(rand(Float32, N, 4));
ycont = 2*X.x1 - X.x3 + 0.1*rand(N)
m, M = minimum(ycont), maximum(ycont)
_, a, b, _ = range(m, stop=M, length=4) |> collect
y = map(ycont) do η
    if η < 0.9*a
        'a'
    elseif η < 1.1*b
        'b'
    else
        'c'
    end
end |> categorical;

# TODO: replace Short2 -> Short when
# https://github.com/FluxML/Flux.jl/issues/1372 is resolved:
builder = Short2()
optimiser = Flux.Optimise.Adam(0.03)

losses = []

@testset_accelerated "NeuralNetworkClassifier" accel begin
    Random.seed!(123)
    # Table input:
    basictest(MLJFlux.NeuralNetworkClassifier,
              X,
              y,
              builder,
              optimiser,
              0.85,
              accel)
    # Matrix input:
    basictest(MLJFlux.NeuralNetworkClassifier,
              matrix(X),
              y,
              builder,
              optimiser,
              0.85,
              accel)

    train, test = MLJBase.partition(1:N, 0.7)

    # baseline loss (predict constant probability distribution):
    dict = StatsBase.countmap(y[train])
    prob_given_class = Dict{CategoricalArrays.CategoricalValue,Float64}()
    for (k, v) in dict
        prob_given_class[k] = dict[k]/length(train)
    end
    dist = MLJBase.UnivariateFinite(prob_given_class)
    loss_baseline =
        MLJBase.cross_entropy(fill(dist, length(test)), y[test]) |> mean

    # check flux model is an improvement on predicting constant
    # distribution:
    stable_rng = StableRNGs.StableRNG(123)
    model = MLJFlux.NeuralNetworkClassifier(epochs=50,
                                            builder=builder,
                                            optimiser=optimiser,
                                            acceleration=accel,
                                            batch_size=10,
                                            rng=stable_rng)
    @time mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    first_last_training_loss = MLJBase.report(mach)[1][[1, end]]
    push!(losses, first_last_training_loss[2])
    yhat = MLJBase.predict(mach, rows=test);
    @test mean(MLJBase.cross_entropy(yhat, y[test])) < 0.95*loss_baseline

    optimisertest(MLJFlux.NeuralNetworkClassifier,
                  X,
                  y,
                  builder,
                  optimiser,
                  accel)

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@test all(x->abs(x - reference)/reference < 1e-5, losses[2:end])

true
