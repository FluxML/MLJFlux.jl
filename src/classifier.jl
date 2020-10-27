mutable struct NeuralNetworkClassifier{B,F,O,L} <: MLJModelInterface.Probabilistic
    builder::B
    finaliser::F
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource       # Train on gpu
end

NeuralNetworkClassifier(; builder::B   = Short()
              , finaliser::F = Flux.softmax
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.crossentropy
              , epochs       = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining = false
              , acceleration = CPU1()
              ) where {B,F,O,L} =
                  NeuralNetworkClassifier{B,F,O,L}(builder
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

function MLJModelInterface.fit(model::NeuralNetworkClassifier,
                               verbosity::Int,
                               X,
                               y)

    # (No categorical features)
    n_input = Tables.schema(X).names |> length
    levels = MLJModelInterface.classes(y[1])
    n_output = length(levels)
    chain = Flux.Chain(build(model.builder, n_input, n_output),
                       model.finaliser)

    data = collate(model, X, y)
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

    cache = (deepcopy(model), data, history, n_input, n_output)
    fitresult = (chain, levels)
    report = (training_losses=history, )

    return fitresult, cache, report
end

function MLJModelInterface.predict(model::NeuralNetworkClassifier,
                                   fitresult,
                                   Xnew_)
    chain, levels = fitresult
    Xnew = MLJModelInterface.matrix(Xnew_) |> Mover(model.acceleration)
    probs = vcat([chain(Xnew[i, :])' for i in 1:size(Xnew, 1)]...) |> Flux.cpu
    return MLJModelInterface.UnivariateFinite(levels, probs)
end

function MLJModelInterface.update(model::NeuralNetworkClassifier,
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
        chain = Flux.Chain(build(model.builder, n_input, n_output),
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

MLJModelInterface.fitted_params(::NeuralNetworkClassifier, fitresult) =
    (chain=fitresult[1],)


MLJModelInterface.metadata_model(NeuralNetworkClassifier,
                                 input=Table(Continuous),
                                 target=AbstractVector{<:Finite},
                                 path="MLJFlux.NeuralNetworkClassifier",
                                 descr="A neural network model for making "*
                                 "probabilistic predictions of a "*
                                 "`Multiclass` or `OrderedFactor` target, "*
                                 "given a table of `Continuous` features. ")
