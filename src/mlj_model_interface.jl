# # EQUALITY

# to address #124 and #129:
MLJModelInterface.deep_properties(::Type{<:MLJFluxModel}) =
    (:optimiser, :builder)


# # CLEAN METHOD

const ERR_BAD_OPTIMISER = ArgumentError(
    "Flux.jl optimiser detected. Only optimisers from Optimisers.jl are supported. "*
    "For example, use `optimiser=Optimisers.Momentum()` after `import Optimisers`. "
)

function MLJModelInterface.clean!(model::MLJFluxModel)
    warning = ""
    if model.lambda < 0
        warning *= "Need `lambda ≥ 0`. Resetting `lambda = 0`. "
        model.lambda = 0
    end
    if model.alpha < 0 || model.alpha > 1
        warning *= "Need alpha in the interval `[0, 1]`. "*
            "Resetting `alpha = 0`. "
        model.alpha = 0
    end
    if model.epochs < 0
        warning *= "Need `epochs ≥ 0`. Resetting `epochs = 10`. "
        model.epochs = 10
    end
    if model.batch_size <= 0
        warning *= "Need `batch_size > 0`. Resetting `batch_size = 1`. "
        model.batch_size = 1
    end
    if model.acceleration isa CUDALibs && gpu_isdead()
        warning *= "`acceleration isa CUDALibs` "*
            "but no CUDA device (GPU) currently live. "
    end
    if !(model.acceleration isa CUDALibs || model.acceleration isa CPU1)
        warning *= "`Undefined acceleration, falling back to CPU`"
        model.acceleration = CPU1()
    end
    if model.acceleration isa CUDALibs && model.rng isa Integer
        warning *= "Specifying an RNG seed when "*
            "`acceleration isa CUDALibs()` may fail for layers depending "*
            "on an RNG during training, such as `Dropout`. Consider using "*
            " `Random.default_rng()` instead. `"
    end
    # TODO: This could be removed in next breaking release (0.6.0):
    model.optimiser isa Flux.Optimise.AbstractOptimiser && throw(ERR_BAD_OPTIMISER)

    return warning
end


# # FIT AND  UPDATE

const ERR_BUILDER =
    "Builder does not appear to build an architecture compatible with supplied data. "

true_rng(model) = model.rng isa Integer ? Random.Xoshiro(model.rng) : model.rng

# Models implement L1/L2 regularization by chaining the chosen optimiser with weight/sign
# decay.  Note that the weight/sign decay must be scaled down by the number of batches to
# ensure penalization over an epoch does not scale with the choice of batch size; see
# https://github.com/FluxML/MLJFlux.jl/issues/213.

function regularized_optimiser(model, nbatches)
    model.lambda == 0 && return model.optimiser
    λ_L1 = model.alpha*model.lambda
    λ_L2 = (1 - model.alpha)*model.lambda
    λ_sign = λ_L1/nbatches
    λ_weight = 2*λ_L2/nbatches

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
   else  return Optimisers.OptimiserChain(
        Optimisers.SignDecay(λ_sign),
        Optimisers.WeightDecay(λ_weight),
        model.optimiser,
        )
    end
end

function MLJModelInterface.fit(model::MLJFluxModel,
                               verbosity,
                               X,
                               y)

    move = Mover(model.acceleration)

    rng = true_rng(model)
    shape = MLJFlux.shape(model, X, y)

    chain = try
        build(model, rng, shape) |> move
    catch ex
        @error ERR_BUILDER
        rethrow()
    end

    data = move.(collate(model, X, y, verbosity))
    x = data[1][1]

    try
        chain(x)
    catch ex
        @error ERR_BUILDER
        throw(ex)
    end

    nbatches = length(data[2])
    regularized_optimiser = MLJFlux.regularized_optimiser(model, nbatches)
    optimiser_state = Optimisers.setup(regularized_optimiser, chain)

    chain, optimiser_state, history = train(
        model,
        chain,
        regularized_optimiser,
        optimiser_state,
        model.epochs,
        verbosity,
        data[1],
        data[2],
    )

    cache = (
        deepcopy(model),
        data,
        history,
        shape,
        regularized_optimiser,
        optimiser_state,
        deepcopy(rng),
        move,
    )
    fitresult = MLJFlux.fitresult(model, Flux.cpu(chain), y)

    report = (training_losses=history, )

    return fitresult, cache, report
end

function MLJModelInterface.update(model::MLJFluxModel,
                                  verbosity,
                                  old_fitresult,
                                  old_cache,
                                  X,
                                  y)

    old_model, data, old_history, shape, regularized_optimiser,
        optimiser_state, rng, move = old_cache
    old_chain = old_fitresult[1]

    optimiser_flag = model.optimiser_changes_trigger_retraining &&
        model.optimiser != old_model.optimiser

    keep_chain = !optimiser_flag && model.epochs >= old_model.epochs &&
        MLJModelInterface.is_same_except(model, old_model, :optimiser, :epochs)

    if keep_chain
        chain = move(old_chain)
        epochs = model.epochs - old_model.epochs
        # (`optimiser_state` is not reset)
    else
        move = Mover(model.acceleration)
        rng = true_rng(model)
        chain = build(model, rng, shape) |> move
        # reset `optimiser_state`:
        data = move.(collate(model, X, y))
        nbatches = length(data[2])
        regularized_optimiser = MLJFlux.regularized_optimiser(model, nbatches)
        optimiser_state = Optimisers.setup(regularized_optimiser, chain)
        epochs = model.epochs
    end

    chain, optimiser_state, history = train(
        model,
        chain,
        regularized_optimiser,
        optimiser_state,
        epochs,
        verbosity,
        data[1],
        data[2],
    )
    if keep_chain
        # note: history[1] = old_history[end]
        history = vcat(old_history[1:end-1], history)
    end

    fitresult = MLJFlux.fitresult(model, Flux.cpu(chain), y)
    cache = (
        deepcopy(model),
        data,
        history,
        shape,
        regularized_optimiser,
        optimiser_state,
        deepcopy(rng),
        move,
    )
    report = (training_losses=history, )

    return fitresult, cache, report

end

MLJModelInterface.fitted_params(::MLJFluxModel, fitresult) =
    (chain=fitresult[1],)


# # SUPPORT FOR MLJ ITERATION API

# traits:
MLJModelInterface.supports_training_losses(::Type{<:MLJFluxModel}) =
    true
MLJModelInterface.iteration_parameter(model::Type{<:MLJFluxModel}) = :epochs

# method:
MLJModelInterface.training_losses(::MLJFluxModel, report) =
    report.training_losses[2:end] # exclude pre-training loss
