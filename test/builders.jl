# to control chain initialization:
myinit(n, m) = reshape(convert(Vector{Float32}, (1:n*m)), n , m)

mutable struct TESTBuilder <: MLJFlux.Builder end
MLJFlux.build(builder::TESTBuilder, rng, n_in, n_out) =
    Flux.Chain(Flux.Dense(n_in, n_out, init=myinit))

@testset_accelerated "issue #152" accel begin

    # data:
    n = 100
    d = 5
    Xmat = rand(Float32, n, d)
#    Xmat = fill(one(Float32), n, d)
    X = MLJBase.table(Xmat);
    y = X.x1 .^2 + X.x2 .* X.x3 - 4 * X.x4

    # train a model on all the data using batch size > 1:
    model = MLJFlux.NeuralNetworkRegressor(builder=TESTBuilder(),
                                           batch_size=25,
                                           epochs=1,
                                           loss=Flux.mse,
                                           acceleration=accel)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    # extract the pre-training loss computed in the `fit!(chain, ...)` method:
    pretraining_loss = report(mach).training_losses[1]

    # compute by hand:
    chain0 = myinit(1, d)
    pretraining_yhat = Xmat*chain0' |> vec
    @test y isa Vector && pretraining_yhat isa Vector
    pretraining_loss_by_hand =  MLJBase.l2(pretraining_yhat, y) |> mean
    mean(((pretraining_yhat - y).^2)[1:2])

    # compare:
    @test pretraining_loss ≈ pretraining_loss_by_hand

end

@testset_accelerated "Short" accel begin
    builder = MLJFlux.Short(n_hidden=4, σ=Flux.relu, dropout=0)
    chain = MLJFlux.build(builder, StableRNGs.StableRNG(123), 5, 3)
    ps = Flux.params(chain)
    @test size.(ps) == [(4, 5), (4,), (3, 4), (3,)]

    # reproducibility (without dropout):
    chain2 = MLJFlux.build(builder, StableRNGs.StableRNG(123), 5, 3)
    x = rand(5)
    @test chain(x) ≈ chain2(x)
end
