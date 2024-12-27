### EntityEmbedder with MLJ Interface

# 1. Interface Struct
mutable struct EntityEmbedder{M <: MLJFluxModel} <: Unsupervised
    model::M
end;


# 2. Constructor
function EntityEmbedder(model;)
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
)

MMI.metadata_model(
    EntityEmbedder,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.EntityEmbedder",
)

MMI.target_in_fit(::Type{<:EntityEmbedder}) = true





"""
$(MMI.doc_header(EntityEmbedder))

`EntityEmbedder` implements entity embeddings as in the "Entity Embeddings of Categorical Variables" paper by Cheng Guo, Felix Berkhahn.

# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

    mach = machine(model, X, y)

Here:


- `X` is any table of input features (eg, a `DataFrame`). Features to be transformed must
   have element scitype `Multiclass` or `OrderedFactor`. Use `schema(X)` to 
   check scitypes. 

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous` or `Count` for regression problems and 
  `Multiclass` or `OrderedFactor` for classification problems; check the scitype with `schema(y)`

Train the machine using `fit!(mach)`.

# Hyper-parameters

- `model`: The underlying deep learning model to be used for entity embedding. So far this supports `NeuralNetworkClassifier`, `NeuralNetworkRegressor`, and `MultitargetNeuralNetworkRegressor`.

# Operations

- `transform(mach, Xnew)`: Transform the categorical features of `Xnew` into dense `Continuous` vectors using the trained `MLJFlux.EntityEmbedderLayer` layer present in the network.
    Check relevant documentation [here](https://fluxml.ai/MLJFlux.jl/dev/) and in particular, the `embedding_dims` hyperparameter.


# Examples

```julia
using MLJFlux
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
[`TargetEncoder`](@ref)
"""
EntityEmbedder