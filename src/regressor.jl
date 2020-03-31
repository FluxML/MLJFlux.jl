mutable struct NeuralNetworkRegressor{B<:Builder,O,L} <: MLJModelInterface.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int     # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

NeuralNetworkRegressor(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.mse
              , epochs       = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining=false
              ) where {B,O,L} =
                  NeuralNetworkRegressor{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , epochs
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)


mutable struct MultitargetNeuralNetworkRegressor{B<:Builder,O,L} <: MLJModelInterface.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

MultitargetNeuralNetworkRegressor(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.mse
              , epochs       = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining=false
              ) where {B,O,L} =
                  MultitargetNeuralNetworkRegressor{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , epochs
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)

const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}

function nrows(X)
    Tables.istable(X) || throw(ArgumentError)
    Tables.columnaccess(X) || return length(collect(X))
    # if has columnaccess
    cols = Tables.columntable(X)
    !isempty(cols) || return 0
    return length(cols[1])
end
nrows(y::AbstractVector) = length(y)

function collate(model::Regressor, X, y, batch_size)

    row_batches = Base.Iterators.partition(1:nrows(y), batch_size)

    Xmatrix = MLJModelInterface.matrix(X)'
    if Tables.istable(y)
        ymatrix = MLJModelInterface.matrix(y)'
        return [(Xmatrix[:, b], ymatrix[:, b]) for b in row_batches]
    else
        ymatrix = reduce(hcat, [[tup...] for tup in y])
        return [((Xmatrix[:, b]), ymatrix[b]) for b in row_batches]
    end
end

function MLJModelInterface.fit(model::Regressor, verbosity::Int, X, y)

    # When it has no categorical features
    n_input = Tables.schema(X).names |> length
    data = collate(model, X, y, model.batch_size)

    target_is_multivariate = Tables.istable(y)
    if target_is_multivariate
        target_columns = Tables.schema(y).names
    else
        target_columns = [""] # We won't be using this
    end

    n_output = length(target_columns)
    chain = fit(model.builder, n_input, n_output)

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss,
                          model.epochs, model.batch_size,
                          model.lambda, model.alpha,
                          verbosity, data)

    cache = (deepcopy(model), data, history)
    fitresult = (chain, target_is_multivariate, target_columns)
    report = (training_losses=history,)

    return fitresult, cache, report

end

function MLJModelInterface.predict(model::Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor},
         fitresult, Xnew_)

    chain , ismulti, target_columns = fitresult

    Xnew_ = MLJModelInterface.matrix(Xnew_)

    if ismulti
        ypred = [map(x->x.data, chain(values.(Xnew_[i, :]))) for i in 1:size(Xnew_, 1)]
        return MLJModelInterface.table(reduce(hcat, y for y in ypred)', names=target_columns)
    else
        return [chain(values.(Xnew_[i, :]))[1] for i in 1:size(Xnew_, 1)]
    end
end

function MLJModelInterface.update(model::Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor},
             verbosity::Int, old_fitresult, old_cache, X, y)

    old_model, data, old_history = old_cache
    old_chain, target_is_multivariate, target_columns = old_fitresult

    keep_chain =  model.epochs >= old_model.epochs &&
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
        epochs = model.epochs - old_model.epochs
    else
        n_input = Tables.schema(X).names |> length
        if target_is_multivariate
            target_columns = Tables.schema(y).names
        else
            target_columns = [""] # We won't be using this
        end
        n_output = length(target_columns)
        chain = fit(model.builder, n_input, n_output)
        data = collate(model, X, y, model.batch_size)
        epochs = model.epochs
    end

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss, epochs,
                                model.batch_size, model.lambda, model.alpha,
                                verbosity, data)
    if keep_chain
        history = vcat(old_history, history)
    end
    fitresult = (chain, target_is_multivariate, target_columns)
    cache = (deepcopy(model), data, history)
    report = (training_losses=history,)

    return fitresult, cache, report

end

MLJModelInterface.metadata_model(NeuralNetworkRegressor,
               input=MLJModelInterface.Table(MLJModelInterface.Continuous),
               target=AbstractVector{<:MLJModelInterface.Continuous},
               path="MLJFlux.NeuralNetworkRegressor",
               descr = "A neural network model for making deterministic predictions of a
               `Continuous` multi-target, presented as a table,  given a table of `Continuous` features. ")


MLJModelInterface.metadata_model(MultitargetNeuralNetworkRegressor,
               input=MLJModelInterface.Table(MLJModelInterface.Continuous),
               target=MLJModelInterface.Table(MLJModelInterface.Continuous),
               path="MLJFlux.NeuralNetworkRegressor",
               descr = "A neural network model for making deterministic predictions of a
                    `Continuous` mutli-target, presented as a table,  given a table of `Continuous` features.")
