mutable struct ImageClassifier{B<:MLJFlux.Builder,O,L} <: MLJModelInterface.Probabilistic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end


ImageClassifier(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.crossentropy
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining = false) where {B,O,L} =
                  ImageClassifier{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)


function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = Flux.onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

# This will not only group into batches, but also convert to Flux compatible tensors
function collate(model::ImageClassifier, X, Y, batch_size)
    mb_idxs = partition(1:length(X), batch_size)
    return [make_minibatch(X, Y, i) for i in mb_idxs]
end

function MLJModelInterface.fit(model::ImageClassifier, verbosity::Int, X_, y_)

    data = collate(model, X_, y_, model.batch_size)

    target_is_multivariate = y_ isa AbstractVector{<:Tuple}



    a_target_element = first(y_)
    levels = MLJModelInterface.classes(a_target_element)
    n_output = length(levels)
    n_input = size(X_[1])
    chain = fit(model.builder,n_input, n_output)

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss,
        model.n, model.batch_size,
        model.lambda, model.alpha,
        verbosity, data)

    cache = deepcopy(model), data, history
    fitresult = (chain, target_is_multivariate, levels)

    report = (training_losses=[loss.data for loss in history])

    return fitresult, cache, report
end

function MLJModelInterface.predict(model::ImageClassifier, fitresult, Xnew)
    chain = fitresult[1]
    ismulti = fitresult[2]
    levels = fitresult[3]
    return [MLJModelInterface.UnivariateFinite(levels, Flux.softmax(chain(Float64.(Xnew[i])).data)) for i in 1:length(Xnew)]

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
               input=MLJModelInterface.Table(MLJModelInterface.Continuous),
               target=AbstractVector{<:MLJModelInterface.GrayImage},
               path="MLJFlux.ImageClassifier",
               descr = descr="A neural network model for making probabilistic predictions of a `GreyImage` target,
                given a table of `Continuous` features. ")
