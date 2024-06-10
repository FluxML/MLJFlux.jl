Base.@deprecate fit!(model::MLJFlux.MLJFluxModel, penalty, chain, optimiser, epochs, verbosity, X, y) train(
    model::MLJFlux.MLJFluxModel,
    chain,
    optimiser,
    optimiser_state,
    epochs,
    verbosity,
    X,
    y,
) false

Base.@deprecate train!(model::MLJFlux.MLJFluxModel, penalty, chain, optimiser, X, y) train_epoch(
    model::MLJFlux.MLJFluxModel,
    chain,
    optimiser,
    optimiser_state,
    X,
    y,
) false