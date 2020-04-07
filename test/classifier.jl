## NEURAL NETWORK CLASSIFIER

@testset "NeuralNetworkClassifier" begin
    seed!(1234)
    N = 300
    X = MLJBase.table(rand(Float32, N, 4));
    ycont = 2*X.x1 - X.x3 + 0.1*rand(N)
    m, M = minimum(ycont), maximum(ycont)
    _, a, b, _ = range(m, M, length=4) |> collect
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

    basictest(MLJFlux.NeuralNetworkClassifier, X, y, builder, optimiser)

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
    # distributioin:
    model = MLJFlux.NeuralNetworkClassifier(epochs=150)
    mach = fit!(machine(model, X, y), rows=train, verbosity=0)
    yhat = MLJBase.predict(mach, rows=test);
    @test mean(MLJBase.cross_entropy(yhat, y[test])) < 0.9*loss_baseline
end


## OLD CLASSIFIER TEST

# train = 1:7N
# test = (7N+1):10N


# builder = MLJFlux.Linear(σ=Flux.sigmoid)
# model = MLJFlux.NeuralNetworkClassifier(loss=Flux.crossentropy,
#                                         builder=builder)

# fitresult, cache, report =
#     MLJBase.fit(model, 2, MLJBase.selectrows(X,train), y[train])

# yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

# # Update without completely retraining
# model.epochs = 15
# fitresult, cache, report =
#     MLJBase.update(model, 2, fitresult, cache,
#                    MLJBase.selectrows(X,train), y[train])

# # Update by completely retraining
# model.batch_size = 5
# fitresult, cache, report =
#     MLJBase.update(model, 3, fitresult, cache,
#                    MLJBase.selectrows(X,train), y[train])


true
