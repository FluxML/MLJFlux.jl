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
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
    is_wrapper = true
)

MMI.metadata_model(
    EntityEmbedder,
    load_path = "MLJTransforms.EntityEmbedder",
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
    EntityEmbedder(; model=mljflux_neural_model)

`EntityEmbedder` implements entity embeddings as in the "Entity Embeddings of Categorical Variables" paper by Cheng Guo, Felix Berkhahn.

# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

    mach = machine(model, X, y)

Here:


- `X` is any table of input features supported by the model being wrapped. Features to be transformed must
   have element scitype `Multiclass` or `OrderedFactor`. Use `schema(X)` to 
   check scitypes. 

- `y` is the target, which can be any `AbstractVector` supported by the model being wrapped.

Train the machine using `fit!(mach)`.

# Hyper-parameters

- `model`: The supervised MLJFlux neural network model to be used for entity embedding. 
  This must be one of these: `MLJFlux.NeuralNetworkClassifier`, `NeuralNetworkBinaryClassifier`,
  `MLJFlux.NeuralNetworkRegressor`,`MLJFlux.MultitargetNeuralNetworkRegressor`. The selected model may have hyperparameters 
  that may affect embedding performance, the most notable of which could be the `builder` argument.

# Operations

- `transform(mach, Xnew)`: Transform the categorical features of `Xnew` into dense `Continuous` vectors using the trained `MLJFlux.EntityEmbedderLayer` layer present in the network.
    Check relevant documentation [here](https://fluxml.ai/MLJFlux.jl/dev/) and in particular, the `embedding_dims` hyperparameter.


# Examples

```julia
using MLJ
using CategoricalArrays

# Setup some data
N = 200
X = (;
    Column1 = repeat(Float32[1.0, 2.0, 3.0, 4.0, 5.0], Int(N / 5)),
    Column2 = categorical(repeat(['a', 'b', 'c', 'd', 'e'], Int(N / 5))),
    Column3 = categorical(repeat(["b", "c", "d", "f", "f"], Int(N / 5)), ordered = true),
    Column4 = repeat(Float32[1.0, 2.0, 3.0, 4.0, 5.0], Int(N / 5)),
    Column5 = randn(Float32, N),
    Column6 = categorical(
        repeat(["group1", "group1", "group2", "group2", "group3"], Int(N / 5)),
    ),
)
y = categorical([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])           # Classification

# Initiate model
EntityEmbedder = @load EntityEmbedder pkg=MLJFlux
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

clf = NeuralNetworkClassifier(embedding_dims=Dict(:Column2 => 2, :Column3 => 2))

emb = EntityEmbedder(clf)

# Construct machine
mach = machine(emb, X, y)

# Train model
fit!(mach)

# Transform data using model to encode categorical columns
Xnew = transform(mach, X)
Xnew
```

See also
[`NeuralNetworkClassifier`, `NeuralNetworkRegressor`](@ref)
"""
EntityEmbedder