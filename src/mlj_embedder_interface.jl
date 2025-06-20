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
    EntityEmbedder(; model=mljflux_neural_model)

`EntityEmbedder` implements entity embeddings as in the "Entity Embeddings of Categorical Variables" paper by Cheng Guo, Felix Berkhahn.

# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

    mach = machine(embed_model, X, y)

Here:

- `embed_model` is an instance of `EntityEmbedder`, which wraps a supervised MLJFlux model. 
  The supervised model must be one of these: `MLJFlux.NeuralNetworkClassifier`, `NeuralNetworkBinaryClassifier`,
  `MLJFlux.NeuralNetworkRegressor`,`MLJFlux.MultitargetNeuralNetworkRegressor`.

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
y = categorical(repeat(["class1", "class2", "class3", "class4", "class5"], Int(N / 5)))

# Load the entity embedder, it's neural network backbone and the SVC which inherently supports
# only continuous features
EntityEmbedder = @load EntityEmbedder pkg=MLJFlux   
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
SVC = @load SVC pkg=LIBSVM              


emb = EntityEmbedder(NeuralNetworkClassifier(embedding_dims=Dict(:Column2 => 2, :Column3 => 2)))
clf = SVC(cost = 1.0)

pipeline = emb |> clf

# Construct machine
mach = machine(pipeline, X, y)

# Train model
fit!(mach)

# Predict
yhat = predict(mach, X)

# Transform data using model to encode categorical columns
machy = machine(emb, X, y)
fit!(machy)
julia> Xnew = transform(machy, X)
(Column1 = Float32[1.0, 2.0, 3.0, … ],
 Column2_1 = Float32[1.2, 0.08, -0.09, -0.2, 0.94, 1.2,  … ],
 Column2_2 = Float32[-0.87, -0.34, -0.8, 1.6, 0.75, -0.87,  …],
 Column3_1 = Float32[-0.0, 1.56, -0.48, -0.9, -0.9, -0.0, …],
 Column3_2 = Float32[-1.0, 1.1, -1.54, 0.2, 0.2, -1.0,  … ],
 Column4 = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, … ],
 Column5 = Float32[0.27, 0.12, -0.60, 1.5, -0.6, -0.123, … ],
 Column6_1 = Float32[-0.99, -0.99, 0.8, 0.8, 0.34, -0.99, … ],
 Column6_2 = Float32[-1.00, -1.0, 0.19, 0.19, 1.7, -1.00, … ])
```

See also
[`NeuralNetworkClassifier`, `NeuralNetworkRegressor`](@ref)
"""
EntityEmbedder
