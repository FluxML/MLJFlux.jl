# # EQUALITY

# to address #124 and #129:
MLJModelInterface.deep_properties(::Type{<:MLJFluxModel}) =
    (:optimiser, :builder)


# # CLEAN METHOD

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
    if ! (model.acceleration isa CUDALibs || model.acceleration isa CPU1)
        warning *= "`Undefined acceleration, falling back to CPU`"
        model.acceleration = CPU1()
    end
    return warning
end


# # FIT AND  UPDATE

const ERR_BUILDER = 
    "Builder does not appear to build an architecture compatible with supplied data. "

true_rng(model) = model.rng isa Integer ? MersenneTwister(model.rng) : model.rng

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
    end
    
    penalty = Penalty(model)
    data = move.(collate(model, X, y))

    x = data |> first |> first
    try
        chain(x)
    catch ex
        @error ERR_BUILDER
        throw(ex)
    end 

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(model,
                          penalty,
                          chain,
                          optimiser,
                          model.epochs,
                          verbosity,
                          data[1],
                          data[2])

    # `optimiser` is now mutated

    cache = (deepcopy(model),
             data,
             history,
             shape,
             optimiser,
             deepcopy(rng),
             move)
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

    old_model, data, old_history, shape, optimiser, rng, move = old_cache
    old_chain = old_fitresult[1]

    optimiser_flag = model.optimiser_changes_trigger_retraining &&
        model.optimiser != old_model.optimiser

    keep_chain = !optimiser_flag && model.epochs >= old_model.epochs &&
        MLJModelInterface.is_same_except(model, old_model, :optimiser, :epochs)

    if keep_chain
        chain = move(old_chain)
        epochs = model.epochs - old_model.epochs
    else
        move = Mover(model.acceleration)
        rng = true_rng(model)
        chain = build(model, rng, shape) |> move
        data = move.(collate(model, X, y))
        epochs = model.epochs
    end

    penalty = Penalty(model)

    # we only get to keep the optimiser "state" carried over from
    # previous training if we're doing a warm restart and the user has not
    # changed the optimiser hyper-parameter:
    if !keep_chain ||
        !MLJModelInterface._equal_to_depth_one(model.optimiser,
                                              old_model.optimiser)
        optimiser = deepcopy(model.optimiser)
    end

    chain, history = fit!(model,
                          penalty,
                          chain,
                          optimiser,
                          epochs,
                          verbosity,
                          data[1],
                          data[2])
    if keep_chain
        # note: history[1] = old_history[end]
        history = vcat(old_history[1:end-1], history)
    end

    fitresult = MLJFlux.fitresult(model, Flux.cpu(chain), y)
    cache = (deepcopy(model),
             data,
             history,
             shape,
             optimiser,
             deepcopy(rng),
             move)
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
