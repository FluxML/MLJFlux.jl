mutable struct ImageClassifier{B,F,O,L} <: MLJModelInterface.Probabilistic
    builder::B
    finaliser::F
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource
end

function ImageClassifier(; builder::B   = Short()
                         , finaliser::F = Flux.softmax
                         , optimiser::O = Flux.Optimise.ADAM()
                         , loss::L      = Flux.crossentropy
                         , epochs       = 10
                         , batch_size   = 1
                         , lambda       = 0
                         , alpha        = 0
                         , optimiser_changes_trigger_retraining = false
                         , acceleration = CPU1()
                         ) where {B,F,O,L}

    model = ImageClassifier{B,F,O,L}(builder
                                     , finaliser
                                     , optimiser
                                     , loss
                                     , epochs
                                     , batch_size
                                     , lambda
                                     , alpha
                                     , optimiser_changes_trigger_retraining
                                     , acceleration
                                     )

   message = clean!(model)
   isempty(message) || @warn message

    return model
end

function MLJModelInterface.fit(model::ImageClassifier,
                               verbosity::Int,
                               X_,
                               y_)

    data = collate(model, X_, y_)

    levels = MLJModelInterface.classes(y_[1])
    n_output = length(levels)
    n_input = size(X_[1])

    if scitype(first(X_)) <: GrayImage{A, B} where A where B
        n_channels = 1      # 1-D image
    else
        n_channels = 3      # 3-D color image
    end

    chain0 = build(model.builder, n_input, n_output, n_channels)
    chain = Flux.Chain(chain0, model.finaliser)

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain,
                          optimiser,
                          model.loss,
                          model.epochs,
                          model.lambda,
                          model.alpha,
                          verbosity,
                          data,
                          model.acceleration)

    cache = deepcopy(model), data, history, n_input, n_output
    fitresult = (chain, levels)

    report = (training_losses=history, )

    return fitresult, cache, report
end

function MLJModelInterface.predict(model::ImageClassifier, fitresult, Xnew)
    chain, levels = fitresult
    X = reformat(Xnew)
    probs = vcat([chain(X[:,:,:,idx:idx])'
                  for idx in 1:length(Xnew)]...)
    return MLJModelInterface.UnivariateFinite(levels, probs)
end

function MLJModelInterface.update(model::ImageClassifier,
                                  verbosity::Int,
                                  old_fitresult,
                                  old_cache,
                                  X,
                                  y)

    old_model, data, old_history, n_input, n_output = old_cache
    old_chain, levels = old_fitresult

    optimiser_flag = model.optimiser_changes_trigger_retraining &&
        model.optimiser != old_model.optimiser

    keep_chain = !optimiser_flag && model.epochs >= old_model.epochs &&
        MLJModelInterface.is_same_except(model, old_model, :optimiser, :epochs)

    if keep_chain
        chain = old_chain
        epochs = model.epochs - old_model.epochs
    else
        if scitype(first(X)) <: GrayImage{A, B} where A where B
            n_channels = 1      # 1-D image
        else
            n_channels = 3      # 3-D color image
        end
        chain = Flux.Chain(build(model.builder, n_input, n_output, n_channels),
                           model.finaliser)
        data = collate(model, X, y)
        epochs = model.epochs
    end

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain,
                          optimiser,
                          model.loss,
                          epochs,
                          model.lambda,
                          model.alpha,
                          verbosity,
                          data,
                          model.acceleration)
    if keep_chain
        # note: history[1] = old_history[end]
        history = vcat(old_history[1:end-1], history)
    end

    fitresult = (chain, levels)
    cache = (deepcopy(model), data, history, n_input, n_output)
    report = (training_losses=history, )

    return fitresult, cache, report

end

MLJModelInterface.fitted_params(::ImageClassifier, fitresult) =
    (chain=fitresult[1],)

MLJModelInterface.metadata_model(ImageClassifier,
               input=AbstractVector{<:MLJModelInterface.GrayImage},
               target=AbstractVector{<:Multiclass},
               path="MLJFlux.ImageClassifier",
               descr="A neural network model for making probabilistic predictions of a `GrayImage` target,
                given a table of `Continuous` features. ")
