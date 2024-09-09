"""
A file containing functions used in the `fit` and `update` methods in `mlj_model_interface.jl`
"""

# Converts input to table if it's a matrix
convert_to_table(X) = X isa Matrix ? Tables.table(X) : X


# Construct model chain and throws error if it fails
function construct_model_chain(model, rng, shape, move)
    chain = try
        build(model, rng, shape) |> move
    catch ex
        @error ERR_BUILDER
        rethrow()
    end
    return chain
end

# Test whether constructed chain works else throws error
function test_chain_works(x, chain)
    try
        chain(x)
    catch ex
        @error ERR_BUILDER
        throw(ex)
    end
end

# Models implement L1/L2 regularization by chaining the chosen optimiser with weight/sign
# decay.  Note that the weight/sign decay must be scaled down by the number of batches to
# ensure penalization over an epoch does not scale with the choice of batch size; see
# https://github.com/FluxML/MLJFlux.jl/issues/213.

function regularized_optimiser(model, nbatches)
    model.lambda == 0 && return model.optimiser
    λ_L1 = model.alpha * model.lambda
    λ_L2 = (1 - model.alpha) * model.lambda
    λ_sign = λ_L1 / nbatches
    λ_weight = 2 * λ_L2 / nbatches

    # recall components in an optimiser chain are executed from left to right:
    if model.alpha == 0
        return Optimisers.OptimiserChain(
            Optimisers.WeightDecay(λ_weight),
            model.optimiser,
        )
    elseif model.alpha == 1
        return Optimisers.OptimiserChain(
            Optimisers.SignDecay(λ_sign),
            model.optimiser,
        )
    else
        return Optimisers.OptimiserChain(
            Optimisers.SignDecay(λ_sign),
            Optimisers.WeightDecay(λ_weight),
            model.optimiser,
        )
    end
end

# Prepares optimiser for training
function prepare_optimiser(data, model, chain)
    nbatches = length(data[2])
    regularized_optimiser = MLJFlux.regularized_optimiser(model, nbatches)
    optimiser_state = Optimisers.setup(regularized_optimiser, chain)
    return regularized_optimiser, optimiser_state
end