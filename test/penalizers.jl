using Statistics
import MLJFlux
import Flux

@testset  "Penalizer" begin
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

@testset "Penalty" begin
    model = MLJFlux.NeuralNetworkRegressor(lambda=1, alpha=1, loss=Flux.mae)
    chain = Flux.Dense(3, 1, identity)
    w = Flux.params(chain)
    p = MLJFlux.Penalty(model)

    # compare loss by hand and with penalized loss function:
    penalty = (sum(abs.(chain.weight)) + abs(chain.bias[1]))
    @test p(w) ≈ penalty
end
