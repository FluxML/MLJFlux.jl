using Statistics

@testset  "penalties" begin
    A = [-1 2; -3 4]
    lambda = 1

    # 100% L2:
    alpha = 0
    penalty = MLJFlux.Penalizer(lambda, alpha)
    @test penalty(A) ≈ 1 + 4 + 9 + 16

    # 100% L1:
    alpha = 1
    penalty = MLJFlux.Penalizer(lambda, alpha)
    @test penalty(A) ≈ 1 + 2 + 3 + 4

    # no strength:
    lambda = 0
    alpha = 42.324
    penalty = MLJFlux.Penalizer(lambda, alpha)
    @test penalty(A) == 0
end

@testset "penalized_losses" begin
    # construct a penalized loss function:
    model = MLJFlux.NeuralNetworkRegressor(lambda=1, alpha=1, loss=Flux.mae)
    chain = Flux.Dense(3, 1, identity)
    p = MLJFlux.PenalizedLoss(model, chain)

    # construct a batch:
    b = 5
    x = rand(Float32, 3, b)
    y = rand(Float32, 1, b)

    # compare loss by hand and with penalized loss function:
    penalty = (sum(abs.(chain.weight)) + abs(chain.bias[1]))
    yhat = chain(x)
    @test p(x, y) ≈ Flux.mae(yhat, y) + penalty
end
