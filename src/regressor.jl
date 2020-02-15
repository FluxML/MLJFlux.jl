mutable struct NeuralNetworkRegressor{B<:Builder,O,L} <: MLJBase.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

NeuralNetworkRegressor(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.mse
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining=false
              ) where {B,O,L} =
                  NeuralNetworkRegressor{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)


mutable struct MultivariateNeuralNetworkRegressor{B<:Builder,O,L} <: MLJBase.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

MultivariateNeuralNetworkRegressor(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.mse
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining=false
              ) where {B,O,L} =
                  MultivariateNeuralNetworkRegressor{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)


function collate(model::Union{NeuralNetworkRegressor, MultivariateNeuralNetworkRegressor},
                 X, y, batch_size)

    ymatrix = reduce(hcat, [[tup...] for tup in y])
    row_batches = Base.Iterators.partition(1:length(y), batch_size)

    Xmatrix = MLJBase.matrix(X)'
    if y isa AbstractVector{<:Tuple}
        return [(Xmatrix[:, b], ymatrix[:, b]) for b in row_batches]
    else
        return [((Xmatrix[:, b]), ymatrix[b]) for b in row_batches]
    end
end

function MLJBase.fit(model::Union{NeuralNetworkRegressor, MultivariateNeuralNetworkRegressor},
                     verbosity::Int,
                     X, y)

    # When it has no categorical features
    n = MLJBase.schema(X).names |> length
    m = length(y[1])
    chain = fit(model.builder, n, m)

    data = collate(model, X, y, model.batch_size)

    target_is_multivariate = y isa AbstractVector{<:Tuple}

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss,
                          model.n, model.batch_size,
                          model.lambda, model.alpha,
                          verbosity, data)

    cache = (deepcopy(model), data, history)
    fitresult = (chain, target_is_multivariate)
    report = (training_losses=history,)

    return fitresult, cache, report

end

# for univariate targets:
function MLJBase.predict(model::Union{NeuralNetworkRegressor, MultivariateNeuralNetworkRegressor},
         fitresult, Xnew_)

    chain = fitresult[1]
    ismulti = fitresult[2]
    
    Xnew_ = MLJBase.matrix(Xnew_)
    println(ismulti)
    if ismulti
        ypred = [chain(values.(Xnew_[i, :])) for i in 1:size(Xnew_, 1)]
        return reduce(vcat, y for y in ypred)' |> MLJBase.table
    else
        return [chain(values.(Xnew_[i, :]))[1] for i in 1:size(Xnew_, 1)]
    end
end

function MLJBase.update(model::Union{NeuralNetworkRegressor, MultivariateNeuralNetworkRegressor},
             verbosity::Int, old_fitresult, old_cache, X, y)

    old_model, data, old_history = old_cache
    old_chain, target_is_multivariate = old_fitresult

    keep_chain =  model.n >= old_model.n &&
        model.loss == old_model.loss &&
        model.batch_size == old_model.batch_size &&
        model.lambda == old_model.lambda &&
        model.alpha == old_model.alpha &&
        model.builder == old_model.builder &&
        #model.embedding_choice == old_model.embedding_choice &&
        (!model.optimiser_changes_trigger_retraining ||
         model.optimiser == old_model.optimiser)

    if keep_chain
        chain = old_chain
        epochs = model.n - old_model.n
    else
        data = collate(model, X, y, model.batch_size)
        epochs = model.n
    end

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss, epochs,
                                model.batch_size, model.lambda, model.alpha,
                                verbosity, data)
    if keep_chain
        history = vcat(old_history, history)
    end
    fitresult = (chain, target_is_multivariate)
    cache = (deepcopy(model), data, history)
    report = (training_losses=history,)

    return fitresult, cache, report

end

MLJBase.metadata_model(NeuralNetworkRegressor,
               input=MLJBase.Table(MLJBase.Continuous),
               target=AbstractVector{<:MLJBase.Continuous},
               path="MLJFlux.NeuralNetworkRegressor",
               descr = "A neural network model for making deterministic predictions of a 
               `Continuous` multi-target, presented as a table,  given a table of `Continuous` features. ")


MLJBase.metadata_model(MultivariateNeuralNetworkRegressor,
               input=MLJBase.Table(MLJBase.Continuous),
               target=MLJBase.Table(MLJBase.Continuous),
               path="MLJFlux.NeuralNetworkRegressor",
               descr = "A neural network model for making deterministic predictions of a 
                    `Continuous` mutli-target, presented as a table,  given a table of `Continuous` features.")
