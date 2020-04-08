mutable struct ImageClassifier{B<:MLJFlux.Builder,F,O,L} <: MLJModelInterface.Probabilistic
    builder::B
    finalizer::F
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

ImageClassifier(; builder::B   = Linear()
              , finalizer::F = Flux.softmax
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.crossentropy
              , epochs       = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining = false) where {B,O,L} =
                  ImageClassifier{B,O,L}(builder
                                       , finalizer
                                       , optimiser
                                       , loss
                                       , epochs
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)


                                       #=
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = Flux.onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

# This will not only group into batches, but also convert to Flux
# compatible tensors
function collate(model::ImageClassifier, X, Y)
    row_batches = Base.iterators.partition(1:length(X), model.batch_size)
    return [make_minibatch(X, Y, i) for i in row_batches]
end
=#

function MLJModelInterface.fit(model::ImageClassifier, verbosity::Int, X_, y_)

    data = collate(model, X_, y_, model.batch_size)

    a_target_element = first(y_)
    levels = MLJModelInterface.classes(a_target_element)
    n_output = length(levels)
    n_input = size(X_[1])
    chain = Flux.Chain(fit(model.builder,n_input, n_output), model.finalizer)

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss,
        model.n, model.lambda, model.alpha,
        verbosity, data)

    cache = deepcopy(model), data, history
    fitresult = (chain, levels)

    report = (training_losses=[loss.data for loss in history])

    return fitresult, cache, report
end

function MLJModelInterface.predict(model::ImageClassifier, fitresult, Xnew)
    chain, levels = fitresult
    [MLJModelInterface.UnivariateFinite(levels, map(x -> x.data, chain(Xnew[:, :, :, i]))) for i in 1:size(Xnew, 4)]
end

function MLJModelInterface.update(model::ImageClassifier, verbosity::Int, old_fitresult, old_cache, X, y)

    old_model, data, old_history = old_cache
    old_chain, target_is_multivariate = old_fitresult
    levels = old_fitresult[3]

    keep_chain =  model.n >= old_model.n &&
        model.loss == old_model.loss &&
        model.batch_size == old_model.batch_size &&
        model.lambda == old_model.lambda &&
        model.alpha == old_model.alpha &&
        model.builder == old_model.builder &&
        (!model.optimiser_changes_trigger_retraining ||
         model.optimiser == old_model.optimiser)

    if keep_chain
        chain = old_chain
        epochs = model.n - old_model.n
    else
        n_input = Tables.schema(X).names |> length
        n_output = length(levels)
        chain = fit(model.builder, n_input, n_output)
        epochs = model.n
    end

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss, epochs,
                                model.batch_size, model.lambda, model.alpha,
                                verbosity, data)
    if keep_chain
        history = vcat(old_history, history)
    end
    fitresult = (chain, target_is_multivariate, levels)
    cache = (deepcopy(model), data, history)
    report = (training_losses=[loss.data for loss in history])

    return fitresult, cache, report

end

MLJModelInterface.metadata_model(ImageClassifier,
               input=AbstractVector{<:MLJModelInterface.GrayImage},
               target=AbstractVector{<:Multiclass},
               path="MLJFlux.ImageClassifier",
               descr = descr="A neural network model for making probabilistic predictions of a `GreyImage` target,
                given a table of `Continuous` features. ")
