# to control chain initialization:
myinit(n, m) = reshape(float(1:n*m), n , m)

mutable struct TESTBuilder <: MLJFlux.Builder end
MLJFlux.build(builder::TESTBuilder, n_in, n_out) =
    Flux.Chain(Flux.Dense(n_in, n_out, init=myinit))

@testset_accelerated "issue #152" accel begin

    # data:
    n = 100
    d = 5
    Xmat = rand(Float64, n, d)
    X = MLJBase.table(Xmat);
    y = X.x1 .^2 + X.x2 .* X.x3 - 4 * X.x4

    # train a model on all the data using batch size > 1:
    model = MLJFlux.NeuralNetworkRegressor(builder = TESTBuilder(),
                                   batch_size=25,
                                   epochs=1,
                                   loss=Flux.mse)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    # extract the pre-training loss computed in the `fit!(chain, ...)` method:
    pretraining_loss = report(mach).training_losses[1]

    # compute by hand:
    chain0 = myinit(1, d)
    pretraining_yhat = Xmat*chain0' |> vec
    @test y isa Vector && pretraining_yhat isa Vector
    pretraining_loss_by_hand =  MLJBase.l2(pretraining_yhat, y) |> mean

    # compare:
    @test pretraining_loss ≈ pretraining_loss_by_hand

end
