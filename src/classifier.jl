mutable struct NeuralNetworkClassifier{B<:Builder,O,L} <: MLJModelInterface.Probabilistic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

NeuralNetworkClassifier(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.crossentropy
              , epochs       = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining = false
              ) where {B,O,L} =
                  NeuralNetworkClassifier{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , epochs
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining
                                       )


function collate(model::NeuralNetworkClassifier,
                 X, y, batch_size)

    row_batches = Base.Iterators.partition(1:length(y), batch_size)

    levels = y |> first |> MLJModelInterface.classes
    ymatrix = hcat([Flux.onehot(ele, levels) for ele in y]...,)

    Xmatrix = MLJModelInterface.matrix(X)'
    if y isa AbstractVector{<:Tuple}
        return [(Xmatrix[:, b], ymatrix[:, b]) for b in row_batches]
    else
        return [((Xmatrix[:, b]), ymatrix[:, b]) for b in row_batches]
    end
    
end


function MLJModelInterface.fit(model::NeuralNetworkClassifier, verbosity::Int,
                        X, y)
    
    # When it has no categorical features
    n_input = Tables.schema(X).names |> length
    n_output = length(levels(y))
    chain = fit(model.builder, n_input, n_output)

    data = collate(model, X, y, model.batch_size)
    target_is_multivariate = y isa AbstractVector{<:Tuple}

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss,
                          model.epochs, model.batch_size,
                          model.lambda, model.alpha,
                          verbosity, data)

    cache = (deepcopy(model), data, history)
    fitresult = (chain, target_is_multivariate, y[1])
    report = (training_losses=history,)
    return fitresult, cache, report
end

function MLJModelInterface.predict(model::NeuralNetworkClassifier, fitresult, Xnew_)
    chain , ismulti, levels = fitresult
    
    Xnew_ = MLJModelInterface.matrix(Xnew_)

    return [MLJModelInterface.UnivariateFinite(MLJModelInterface.classes(levels), map(x->x.data, Flux.softmax(chain(Xnew_[i, :]))) |> vec) for i in 1:size(Xnew_, 1)]

end

function MLJModelInterface.update(model::NeuralNetworkClassifier, verbosity::Int, old_fitresult, old_cache, X, y)

    old_model, data, old_history = old_cache
    old_chain, target_is_multivariate = old_fitresult

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
        n_output = length(MLJModelInterface.classes(y[1]))
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
    fitresult = (chain, target_is_multivariate, levels(y))
    cache = (deepcopy(model), data, history)
    report = (training_losses=history,)

    return fitresult, cache, report

end

MLJModelInterface.metadata_model(NeuralNetworkClassifier,
               input=MLJModelInterface.Table(MLJModelInterface.Continuous),
               target=AbstractVector{<:MLJModelInterface.Finite},
               path="MLJFlux.NeuralNetworkClassifier",
               descr="A neural network model for making probabilistic predictions of a `Mutliclass` 
               or `OrderedFactor` target, given a table of `Continuous` features. ")

