HERE = @__DIR__
include(joinpath(HERE, "common_workflows", "composition", "generate.jl"))
include(joinpath(HERE, "common_workflows", "comparison", "generate.jl"))
include(joinpath(HERE, "common_workflows", "architecture_search", "generate.jl"))
include(joinpath(HERE, "common_workflows", "hyperparameter_tuning", "generate.jl"))
include(joinpath(HERE, "common_workflows", "entity_embeddings", "generate.jl"))
include(joinpath(HERE, "common_workflows", "incremental_training", "generate.jl"))
include(joinpath(HERE, "common_workflows", "early_stopping", "generate.jl"))
include(joinpath(HERE, "common_workflows", "live_training", "generate.jl"))
include(joinpath(HERE, "extended_examples", "MNIST", "generate.jl"))

@info "Ignore warnings above about Pluto notebooks; we don't provide these"
@info "Where *executed* jupyter notebooks are not generated, check the corresponding "*
    "README to check we are not providing them. Generally we don't provide them where "*
    "Documenter.jl has had trouble generating them in the past. "
