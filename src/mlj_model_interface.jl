# # EQUALITY

# to address #124 and #129:
MLJModelInterface.deep_properties(::Type{<:MLJFluxModel}) =
    (:optimiser, :builder)


# # CLEAN METHOD

const ERR_BAD_OPTIMISER = ArgumentError(
    "Flux.jl optimiser detected. Only optimisers from Optimisers.jl are supported. " *
    "For example, use `optimiser=Optimisers.Momentum()` after `import Optimisers`. ",
)

function MLJModelInterface.clean!(model::MLJFluxModel)
    warning = ""
    if model.lambda < 0
        warning *= "Need `lambda ≥ 0`. Resetting `lambda = 0`. "
        model.lambda = 0
    end
    if model.alpha < 0 || model.alpha > 1
        warning *= "Need alpha in the interval `[0, 1]`. " *
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
        warning *=
            "`acceleration isa CUDALibs` " *
            "but no CUDA device (GPU) currently live. "
    end
    if !(model.acceleration isa CUDALibs || model.acceleration isa CPU1)
        warning *= "`Undefined acceleration, falling back to CPU`"
        model.acceleration = CPU1()
    end
    if model.acceleration isa CUDALibs && model.rng isa Integer
        warning *=
            "Specifying an RNG seed when " *
            "`acceleration isa CUDALibs()` may fail for layers depending " *
            "on an RNG during training, such as `Dropout`. Consider using " *
            " `Random.default_rng()` instead. `"
    end
    # TODO: This could be removed in next breaking release (0.6.0):
    model.optimiser isa Flux.Optimise.AbstractOptimiser && throw(ERR_BAD_OPTIMISER)

    return warning
end


# # FIT AND  UPDATE

const ERR_BUILDER = "Builder does not appear to build an architecture compatible with supplied data. "

true_rng(model) = model.rng isa Integer ? Random.Xoshiro(model.rng) : model.rng


function MLJModelInterface.fit(model::MLJFluxModel,
    verbosity,
    X,
    y)
    # GPU and rng related variables
    move = Mover(model.acceleration)
    rng = true_rng(model)

    # Get input properties
    shape = MLJFlux.shape(model, X, y)
    cat_inds = get_cat_inds(X)
    pure_continuous_input = isempty(cat_inds)

    # Decide whether to enable entity embeddings (e.g., ImageClassifier won't)
    enable_entity_embs = is_embedding_enabled(model) && !pure_continuous_input

    # Prepare entity embeddings inputs and encode X if entity embeddings enabled
    featnames = []
    if enable_entity_embs
        X = convert_to_table(X)
        featnames = Tables.schema(X).names
    end

    # entityprops is (index = cat_inds[i], levels = num_levels[i], newdim = newdims[i]) 
    # for each categorical feature
    default_embedding_dims = enable_entity_embs ? model.embedding_dims : Dict{Symbol, Real}()
    entityprops, entityemb_output_dim =
        prepare_entityembs(X, featnames, cat_inds, default_embedding_dims)
    X, ordinal_mappings = ordinal_encoder_fit_transform(X; featinds = cat_inds)

    ## Construct model chain
    chain =
        (!enable_entity_embs) ? construct_model_chain(model, rng, shape, move) :
        construct_model_chain_with_entityembs(
            model,
            rng,
            shape,
            move,
            entityprops,
            entityemb_output_dim,
        )

    # Format data as needed by Flux and move to GPU 
    data = move.(collate(model, X, y))

    # Test chain works (as it may be custom)
    x = data[1][1]
    test_chain_works(x, chain)

    # Train model with Flux
    regularized_optimiser, optimiser_state =
        prepare_optimiser(data, model, chain)
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

    # Extract embedding matrices
    embedding_matrices = get_embedding_matrices(chain, cat_inds, featnames)

    # Prepare cache for potential warm restarts
    cache = (
        deepcopy(model),
        data,
        history,
        shape,
        regularized_optimiser,
        optimiser_state,
        deepcopy(rng),
        move,
        entityprops,
        entityemb_output_dim,
        ordinal_mappings,
        featnames,
    )

    # Prepare fitresult 
    fitresult =
        MLJFlux.fitresult(model, Flux.cpu(chain), y, ordinal_mappings, embedding_matrices)

    # Prepare report
    report = (training_losses = history,)

    return fitresult, cache, report
end

function MLJModelInterface.update(model::MLJFluxModel,
    verbosity,
    old_fitresult,
    old_cache,
    X,
    y)
    # Decide whether to enable entity embeddings (e.g., ImageClassifier won't)
    cat_inds = get_cat_inds(X)
    pure_continuous_input = (length(cat_inds) == 0)
    enable_entity_embs = is_embedding_enabled(model) && !pure_continuous_input

    # Unpack cache from previous fit
    old_model,
    data,
    old_history,
    shape,
    regularized_optimiser,
    optimiser_state,
    rng,
    move,
    entityprops,
    entityemb_output_dim,
    ordinal_mappings,
    featnames = old_cache
    cat_inds = [prop.index for prop in entityprops]

    # Extract chain
    old_chain = old_fitresult[1]

    # Decide whether optimiser should trigger retraining from scratch
    optimiser_flag =
        model.optimiser_changes_trigger_retraining &&
        model.optimiser != old_model.optimiser

    # Decide whether to retrain from scratch
    keep_chain =
        !optimiser_flag && model.epochs >= old_model.epochs &&
        MLJModelInterface.is_same_except(model, old_model, :optimiser, :epochs)

    # Use old chain if not retraining from scratch or reconstruct and prepare to retrain
    if keep_chain
        chain = move(old_chain)
        epochs = model.epochs - old_model.epochs
        # (`optimiser_state` is not reset)
    else
        move = Mover(model.acceleration)
        rng = true_rng(model)
        X = convert_to_table(X)
        X = ordinal_encoder_transform(X, ordinal_mappings)
        if enable_entity_embs
            chain =
                construct_model_chain_with_entityembs(
                    model,
                    rng,
                    shape,
                    move,
                    entityprops,
                    entityemb_output_dim,
                )
        else
            chain = construct_model_chain(model, rng, shape, move)
        end
        # reset `optimiser_state`:
        data = move.(collate(model, X, y))
        regularized_optimiser, optimiser_state =
            prepare_optimiser(data, model, chain)
        epochs = model.epochs
    end

    # Train model with Flux
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

    # Properly set history
    if keep_chain
        # note: history[1] = old_history[end]
        history = vcat(old_history[1:end-1], history)
    end

    # Extract embedding matrices
    embedding_matrices = get_embedding_matrices(chain, cat_inds, featnames)

    # Prepare cache, fitresult, and report
    cache = (
        deepcopy(model),
        data,
        history,
        shape,
        regularized_optimiser,
        optimiser_state,
        deepcopy(rng),
        move,
        entityprops, entityemb_output_dim, ordinal_mappings, featnames,
    )
    fitresult =
        MLJFlux.fitresult(model, Flux.cpu(chain), y, ordinal_mappings, embedding_matrices)
    report = (training_losses = history,)

    return fitresult, cache, report

end


# Transformer for entity-enabled models
function MLJModelInterface.transform(
    transformer::MLJFluxModel,
    fitresult,
    Xnew,
)
    # if it doesn't have the property its not an entity-enabled model
    is_embedding_enabled(transformer) || return Xnew
    ordinal_mappings, embedding_matrices = fitresult[3:4]
    Xnew = ordinal_encoder_transform(Xnew, ordinal_mappings)
    Xnew_transf = embedding_transform(Xnew, embedding_matrices)
    return Xnew_transf
end

MLJModelInterface.fitted_params(::MLJFluxModel, fitresult) =
    (chain = fitresult[1],)


# # SUPPORT FOR MLJ ITERATION API

# traits:
MLJModelInterface.supports_training_losses(::Type{<:MLJFluxModel}) =
    true
MLJModelInterface.iteration_parameter(model::Type{<:MLJFluxModel}) = :epochs

# method:
MLJModelInterface.training_losses(::MLJFluxModel, report) =
    report.training_losses[2:end] # exclude pre-training loss
