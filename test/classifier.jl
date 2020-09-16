## NEURAL NETWORK CLASSIFIER

seed!(1234)
N = 300
X = MLJBase.table(rand(Float32, N, 4));
ycont = 2*X.x1 - X.x3 + 0.1*rand(N)
m, M = minimum(ycont), maximum(ycont)
_, a, b, _ = range(m, stop=M, length=4) |> collect
y = map(ycont) do η
    if η < 0.9*a
        :a
    elseif η < 1.1*b
        :b
    else
        :c
    end
end |> categorical;

builder = MLJFlux.Short()
optimiser = Flux.Optimise.ADAM(0.01)

optimiser = Flux.Optimise.ADAM(0.01)

@testset_accelerated "NeuralNetworkClassifier" accel begin
    basictest(MLJFlux.NeuralNetworkClassifier,
              X,
              y,
              builder,
              optimiser,
              0.75,
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
    model = MLJFlux.NeuralNetworkClassifier(epochs=150, acceleration=accel)
    mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    yhat = MLJBase.predict(mach, rows=test);
    @test mean(MLJBase.cross_entropy(yhat, y[test])) < 0.9*loss_baseline
end

true
