mutable struct NeuralNetworkRegressor{B,O,L} <: MLJModelInterface.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int     # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource       # To use GPU
end

function NeuralNetworkRegressor(; builder::B   = Linear()
                                , optimiser::O = Flux.Optimise.ADAM()
                                , loss::L      = Flux.mse
                                , epochs       = 10
                                , batch_size   = 1
                                , lambda       = 0
                                , alpha        = 0
                                , optimiser_changes_trigger_retraining=false
                                , acceleration  = CPU1()
                                ) where {B,O,L}

    model = NeuralNetworkRegressor{B,O,L}(builder
                                          , optimiser
                                          , loss
                                          , epochs
                                          , batch_size
                                          , lambda
                                          , alpha
                                          , optimiser_changes_trigger_retraining
                                          , acceleration)

   message = clean!(model)
   isempty(message) || @warn message

    return model
end

mutable struct MultitargetNeuralNetworkRegressor{B,O,L} <: MLJModelInterface.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    epochs::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource
end

function MultitargetNeuralNetworkRegressor(; builder::B   = Linear()
                                           , optimiser::O = Flux.Optimise.ADAM()
                                           , loss::L      = Flux.mse
                                           , epochs       = 10
                                           , batch_size   = 1
                                           , lambda       = 0
                                           , alpha        = 0
                                           , optimiser_changes_trigger_retraining=false
                                           , acceleration = CPU1()
                                           ) where {B,O,L}

    model = MultitargetNeuralNetworkRegressor{B,O,L}(builder
                                                     , optimiser
                                                     , loss
                                                     , epochs
                                                     , batch_size
                                                     , lambda
                                                     , alpha
                                                     , optimiser_changes_trigger_retraining
                                                     , acceleration)

    message = clean!(model)
    isempty(message) || @warn message

    return model
end

const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}

function MLJModelInterface.fit(model::Regressor, verbosity::Int, X, y)

    # (assumes  no categorical features)
    n_input = Tables.schema(X).names |> length
    data = MLJFlux.collate(model, X, y)

    target_is_multivariate = Tables.istable(y)
    if target_is_multivariate
        target_column_names = Tables.schema(y).names
    else
        target_column_names = [""] # We won't be using this
    end

    n_output = length(target_column_names)
    chain = build(model.builder, n_input, n_output)

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain,
                          optimiser,
                          model.loss,
                          model.epochs,
                          model.lambda,
                          model.alpha,
                          verbosity,
                          #data,
                          model.acceleration,
                          data[1],
                          data[2])

    # note: "state" part of `optimiser` is now mutated!

    cache = (deepcopy(model), data, history, n_input, n_output, optimiser)
    fitresult = (chain, target_is_multivariate, target_column_names)
    report = (training_losses=history,)

    return fitresult, cache, report

end

function MLJModelInterface.update(model::Regressor,
                                  verbosity::Int,
                                  old_fitresult,
                                  old_cache,
                                  X,
                                  y)

    # note: the `optimiser` in `old_cache` stores "state" (eg,
    # momentum); the "state" part of the `optimiser` field of `model`
    # and of `old_model` play no role

    old_model, data, old_history, n_input, n_output, optimiser = old_cache
    old_chain, target_is_multivariate, target_column_names = old_fitresult

    optimiser_flag = model.optimiser_changes_trigger_retraining &&
        model.optimiser != old_model.optimiser

    keep_chain = !optimiser_flag && model.epochs >= old_model.epochs &&
        MLJModelInterface.is_same_except(model, old_model, :optimiser, :epochs)

    if keep_chain
        chain = old_chain
        epochs = model.epochs - old_model.epochs
    else
        chain = build(model.builder, n_input, n_output)
        data = collate(model, X, y)
        epochs = model.epochs
    end

    # we only get to keep the optimiser "state" carried over from
    # previous training if we're doing a warm restart and the user has not
    # changed the optimiser hyper-parameter:
    if !keep_chain ||
        !MLJModelInterface._equal_to_depth_one(model.optimiser,
                                              old_model.optimiser)
        optimiser = deepcopy(model.optimiser)
    end

    chain, history = fit!(chain,
                          optimiser,
                          model.loss,
                          epochs,
                          model.lambda,
                          model.alpha,
                          verbosity,
                          #data,
                          model.acceleration,
                          data[1],
                          data[2])
    if keep_chain
        # note: history[1] = old_history[end]
        history = vcat(old_history[1:end-1], history)
    end

    fitresult = (chain, target_is_multivariate, target_column_names)
    cache = (deepcopy(model), data, history, n_input, n_output, optimiser)
    report = (training_losses=history,)

    return fitresult, cache, report

end

function MLJModelInterface.predict(model::Regressor, fitresult, Xnew_)

    chain , target_is_multivariate, target_column_names = fitresult

    Xnew_ = MLJModelInterface.matrix(Xnew_)

    if target_is_multivariate
        ypred = [chain(values.(Xnew_[i, :]))
                 for i in 1:size(Xnew_, 1)]
        return MLJModelInterface.table(reduce(hcat, y for y in ypred)',
                                       names=target_column_names)
    else
        return [chain(values.(Xnew_[i, :]))[1]
                for i in 1:size(Xnew_, 1)]
    end
end

MLJModelInterface.fitted_params(::Regressor, fitresult) = (chain=fitresult[1],)

MLJModelInterface.metadata_model(NeuralNetworkRegressor,
               input=Table(Continuous),
               target=AbstractVector{<:Continuous},
               path="MLJFlux.NeuralNetworkRegressor",
               descr="A neural network model for making "*
                     "deterministic predictions of a "*
                     "`Continuous` target, given a table of "*
                     "`Continuous` features. ")

MLJModelInterface.metadata_model(MultitargetNeuralNetworkRegressor,
               input=Table(Continuous),
               target=Table(Continuous),
               path="MLJFlux.NeuralNetworkRegressor",
               descr = "A neural network model for making "*
                       "deterministic predictions of a "*
                       "`Continuous` multi-target, presented "*
                       "as a table, given a table of "*
                       "`Continuous` features. ")

