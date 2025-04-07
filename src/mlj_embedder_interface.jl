### EntityEmbedder with MLJ Interface

# 1. Interface Struct
mutable struct EntityEmbedder{M <: MLJFluxModel} <: Unsupervised
    model::M
end;


const ERR_MODEL_UNSPECIFIED = ErrorException("You must specify a suitable MLJFlux supervised model, as in `EntityEmbedder(model=...)`. ")
# 2. Constructor
function EntityEmbedder(;model=nothing)
    model === nothing && throw(ERR_MODEL_UNSPECIFIED)
    return EntityEmbedder(model)
end;


# 4. Fitted parameters (for user access)
MMI.fitted_params(::EntityEmbedder, fitresult) = fitresult

# 5. Fit method
function MMI.fit(transformer::EntityEmbedder, verbosity::Int, X, y)
    return MLJModelInterface.fit(transformer.model, verbosity, X, y)
end;


# 6. Transform method
function MMI.transform(transformer::EntityEmbedder, fitresult, Xnew)
    Xnew_transf = MLJModelInterface.transform(transformer.model, fitresult, Xnew)
    return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
    EntityEmbedder,
    package_name = "MLJFlux",
    package_uuid = "094fc8d1-fd35-5302-93ea-dabda2abf845",
    package_url = "https://github.com/FluxML/MLJFlux.jl",
    is_pure_julia = true,
    is_wrapper = true
)

MMI.metadata_model(
    EntityEmbedder,
    load_path = "MLJFlux.EntityEmbedder",
)

MMI.target_in_fit(::Type{<:EntityEmbedder}) = true

# 9. Forwarding traits
MMI.supports_training_losses(::Type{<:EntityEmbedder}) = true


for trait in [
    :input_scitype,
    :output_scitype,
    :target_scitype,
    ]

    quote
        MMI.$trait(::Type{<:EntityEmbedder{M}}) where M = MMI.$trait(M)
    end |> eval
end

# ## Iteration parameter
prepend(s::Symbol, ::Nothing) = nothing
prepend(s::Symbol, t::Symbol) = Expr(:(.), s, QuoteNode(t))
prepend(s::Symbol, ex::Expr) = Expr(:(.), prepend(s, ex.args[1]), ex.args[2])
quote
    MMI.iteration_parameter(::Type{<:EntityEmbedder{M}}) where M =
        prepend(:model, MMI.iteration_parameter(M))
end |> eval

MMI.training_losses(embedder::EntityEmbedder, report) =
    MMI.training_losses(embedder.model, report)


"""
    EntityEmbedder(; model=supervised_mljflux_model)

Wrapper for a MLJFlux supervised model, to convert it to a transformer. Such transformers
are still presented a target variable in training, but they behave as transformers in MLJ
pipelines. They are entity embedding transformers, in the sense of the article, "Entity
Embeddings of Categorical Variables" by Cheng Guo, Felix Berkhahn.

The atomic `model` must be an instance of `MLJFlux.NeuralNetworkClassifier`,
`MLJFlux.NeuralNetworkBinaryClassifier`, `MLJFlux.NeuralNetworkRegressor`, or
`MLJFlux.MultitargetNeuralNetworkRegressor`. Hyperparameters of the atomic model, in
particular `builder` and `embedding_dims`, will effect embedding performance.

The wrapped model is bound to a machine and trained exactly as the wrapped supervised
`model`, and supports the same form of training data. In particular, a training target
must be supplied.

# Examples

In the following example we wrap a `NeuralNetworkClassifier` as an `EntityEmbedder`, so
that it can be used to supply continuously encoded features to a nearest neighbor model,
which does not support categorical features.

```julia
using MLJ

# Setup some data
N = 400
X = (
  a = rand(Float32, N),
  b = categorical(rand("abcde", N)),
  c = categorical(rand("ABCDEFGHIJ", N), ordered = true),
)
y = categorical(rand("YN", N));

# Initiate model
EntityEmbedder = @load EntityEmbedder pkg=MLJFlux

# Flux model to do learn the entity embeddings:
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

# Other supervised model type, requiring `Continuous` features:
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels

# Instantiate the models:
clf = NeuralNetworkClassifier(embedding_dims=Dict(:b => 2, :c => 3))
emb = EntityEmbedder(clf)

# For illustrative purposes, train the embedder on its own:
mach = machine(emb, X, y)
fit!(mach)
Xnew = transform(mach, X)

# And compare feature scitypes:
schema(X)
schema(Xnew)

# Now construct the pipeline:
pipe = emb |> KNNClassifier()

# And train it to make predictions:
mach = machine(pipe, X, y)
fit!(mach)
predict(mach, X)[1:3]
```

It is to be emphasized that the `NeuralNertworkClassifier` is only being used to
learn entity embeddings, not to make predictions, which here are made by
`KNNClassifier()`.

See also
[`NeuralNetworkClassifier`, `NeuralNetworkRegressor`](@ref)
"""
EntityEmbedder
