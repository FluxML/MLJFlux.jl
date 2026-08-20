using Test
import Optimisers
import MLJFlux
import Flux

lambda = 0.2
alpha = 0.3
nbatches = 5
model = MLJFlux.NeuralNetworkRegressor(; lambda, alpha)
chain = Flux.Dense(3=>1)

@testset "setup_regularized_optimiser" begin
    # use `setup_regularized_optimiser` to get optimiser state:
    optimiser_state = MLJFlux.setup_regularized_optimiser(model, nbatches, chain)

    # get optimiser state by hand:
    λ_L1 = alpha * lambda
    λ_L2 = (1 - alpha) * lambda
    λ_sign = λ_L1 / nbatches
    λ_weight = 2 * λ_L2 / nbatches
    optimiser_chain = Optimisers.OptimiserChain(
        Optimisers.SignDecay(λ_sign),
        Optimisers.WeightDecay(λ_weight),
        model.optimiser,
    )
    optimiser_state2 = Optimisers.setup(optimiser_chain, chain)

    @test optimiser_state == optimiser_state2
end

@testset "adjust! smoke test" begin
    state = MLJFlux.setup_regularized_optimiser(model, nbatches, chain)
    model.optimiser = Optimisers.Adam(123.0)
    state2 = MLJFlux.adjust(state, model, nbatches)
    @test state2 == MLJFlux.setup_regularized_optimiser(model, nbatches, chain)
end
