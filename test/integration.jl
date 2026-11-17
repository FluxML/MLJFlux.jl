using StableRNGs
import MLJFlux
using MLJBase
import StatisticalMeasures
using Tables

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

@testset "logic around changes to the learning rate + warm restart" begin
    model = MLJFlux.NeuralNetworkClassifier(
    builder=MLJFlux.Linear(),
    rng=StableRNGs.StableRNG(123),
    epochs=2,
    )
    model3 = deepcopy(model)
    model4 = deepcopy(model)

    # train for 5 epochs, in two steps:
    mach = machine(model3, X, y)
    fit!(mach, verbosity=2)
    model3.epochs += 3
    fit!(mach, verbosity=2)
    losses_new = training_losses(mach)

    # train for 5 epochs in one step:
    model4.epochs += 3
    mach = machine(model4, X, y)
    fit!(mach, verbosity=0)
    @test losses_new ≈ training_losses(mach)
end
