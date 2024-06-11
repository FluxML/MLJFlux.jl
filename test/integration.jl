rng = StableRNGs.StableRNG(123)

table = load_iris()
y, X = unpack(table, ==(:target), _->true, rng=rng)
X = Tables.table(Float32.(Tables.matrix(X)))

@testset_accelerated "regularization has an effect" accel begin

    model = MLJFlux.NeuralNetworkClassifier(acceleration=accel,
                                    builder=MLJFlux.Linear(),
                                    rng=rng)
    model2 = deepcopy(model)
    model3 = deepcopy(model)
    model4 = deepcopy(model)
    model3.lambda = 0.1
    model4.alpha = 0.1 # still no regularization here because `lambda=0`.

    e = evaluate(model, X, y, resampling=Holdout(), measure=StatisticalMeasures.LogLoss())
    loss1 = e.measurement[1]

    e = evaluate(model2, X, y, resampling=Holdout(), measure=StatisticalMeasures.LogLoss())
    loss2 = e.measurement[1]

    e = evaluate(model3, X, y, resampling=Holdout(), measure=StatisticalMeasures.LogLoss())
    loss3 = e.measurement[1]

    e = evaluate(model4, X, y, resampling=Holdout(), measure=StatisticalMeasures.LogLoss())
    loss4 = e.measurement[1]

    @test loss1 ≈ loss2
    @test !(loss1 ≈ loss3)
    @test loss1 ≈ loss4
end
