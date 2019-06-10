
## Experimental code
mutable struct UnivariateNeuralNetworkClassifier{B<:Builder,O,L} <: MLJBase.Probabilistic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

UnivariateNeuralNetworkClassifier(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.crossentropy
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining = false) where {B,O,L} =
                  NeuralNetworkClassifier{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)

input_is_multivariate(::Type{<:NeuralNetworkClassifier}) = false
input_scitype_union(::Type{<:NeuralNetworkClassifier}) = :Image         # change this later
target_scitype_union(::Type{<:NeuralNetworkClassifier}) = MLJBase.Multiclass

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = Flux.onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

# This will not only group into batches, but also convert to Flux compatible tensors
function collate(model::UnivariateNeuralNetworkClssifier, X, Y, batch_size)
    mb_idxs = partition(1:length(X), batch_size)
    return [make_minibatch(X, Y, i) for i in mb_idxs]
end

function MLJBase.fit(model::UnivariateNeuralNetworkClassifier, verbosity::Int, X_, y_)

    data = collate(model, X_, y_, model.batch_size)

    target_is_multivariate = y_ isa AbstractVector{<:Tuple}

    n = MLJBase.schema(X_).names |> length

    a_target_element = first(y_)
    levels = MLJBase.classes(a_target_element)
    m = length(levels)

    chain = fit(model.builder, n, m)

    # fit!(chain,...) mutates optimisers!!
    # MLJ does not allow fit to mutate models. So:
    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss,
        model.n, model.batch_size,
        model.lambda, model.alpha,
        verbosity, data)

    cache = deepcopy(model), data, history 
    fitresult = (chain, target_is_multivariate, levels)

    report = (training_losses=history, )

    return fitresult, cache, report
end
