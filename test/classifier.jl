## NEURAL NETWORK CLASSIFIER


@testset "NeuralNetworkClassifier" begin
    seed!(1234)
    N = 100
    X = MLJBase.table(rand(Float32, N, 4));
    ycont = 2*X.x1 - X.x3 + 0.6*rand(N)
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

    builder = MLJFlux.Linear(σ=Flux.sigmoid)
    optimiser = Flux.Optimise.ADAM(0.001)

    # model = MLJFlux.NeuralNetworkClassifier()
    # MLJBase.fit(model, 3, X, y)

    train, test = MLJBase.partition(1:N, 0.7)

    # uncomment next line when tests below it are working:
    basictest(MLJFlux.NeuralNetworkClassifier, X, y, builder, optimiser)

    # baseline loss (predict constant probability distribution):
    dict = StatsBase.countmap(y[train])
    prob_given_class = Dict{CategoricalArrays.CategoricalValue,Float64}()
    for (k, v) in dict
        prob_given_class[k] = dict[k]/length(train)
    end
    dist = MLJBase.UnivariateFinite(prob_given_class)
    loss_baseline =
        MLJBase.cross_entropy(fill(dist, length(test)), y[test]) |> mean

    # check flux model performance:
    model = MLJFlux.NeuralNetworkClassifier()
    mach = fit!(machine(model, X, y), rows=train, verbosity=3)
    yhat = predict(mach, rows=test)
    @test MLJBase.cross_entropy(yhat, y[test]) < 0.8*loss_baseline
end


## OLD CLASSIFIER TEST

# train = 1:7N
# test = (7N+1):10N


# builder = MLJFlux.Linear(σ=Flux.sigmoid)
# model = MLJFlux.NeuralNetworkClassifier(loss=Flux.crossentropy, builder=builder)

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
