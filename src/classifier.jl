# if `b` is a builder, then `b(model, rng, shape...)` is called to make a
# new chain, where `shape` is the return value of this method:
"""
    shape(model::NeuralNetworkClassifier, X, y)

A private method that returns the shape of the input and output of the model for given
data `X` and `y`.
"""
function MLJFlux.shape(model::NeuralNetworkClassifier, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    levels = CategoricalArrays.levels(y[1])
    n_output = length(levels)
    n_input = Tables.schema(X).names |> length
    return (n_input, n_output)
end
is_embedding_enabled(::NeuralNetworkClassifier) = true

# builds the end-to-end Flux chain needed, given the `model` and `shape`:
MLJFlux.build(
    model::Union{NeuralNetworkClassifier, NeuralNetworkBinaryClassifier},
    rng,
    shape,
) = Flux.Chain(build(model.builder, rng, shape...), model.finaliser)

# returns the model `fitresult` (see "Adding Models for General Use"
# section of the MLJ manual) which must always have the form `(chain,
# metadata)`, where `metadata` is anything extra needed by `predict`:
MLJFlux.fitresult(
    model::Union{NeuralNetworkClassifier, NeuralNetworkBinaryClassifier},
    chain,
    y,
    ordinal_mappings = nothing,
    embedding_matrices = nothing,
) = (chain, levels(y[1]), ordinal_mappings, embedding_matrices)

function MLJModelInterface.predict(
    model::NeuralNetworkClassifier,
    fitresult,
    Xnew,
)
    chain, levels, ordinal_mappings, _ = fitresult
    Xnew = ordinal_encoder_transform(Xnew, ordinal_mappings)        # what if Xnew is a matrix
    X = _f32(reformat(Xnew), 0)
    probs = vcat([chain(tomat(X[:, i]))' for i in 1:size(X, 2)]...)
    return MLJModelInterface.UnivariateFinite(levels, probs)
end


MLJModelInterface.metadata_model(
    NeuralNetworkClassifier,
    input_scitype = Union{AbstractMatrix{Continuous}, Table(Continuous, Finite)},
    target_scitype = AbstractVector{<:Finite},
    load_path = "MLJFlux.NeuralNetworkClassifier",
)

#### Binary Classifier

function MLJFlux.shape(model::NeuralNetworkBinaryClassifier, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    n_input = Tables.schema(X).names |> length
    return (n_input, 1) # n_output is always 1 for a binary classifier
end
is_embedding_enabled(::NeuralNetworkBinaryClassifier) = true

function MLJModelInterface.predict(
    model::NeuralNetworkBinaryClassifier,
    fitresult,
    Xnew,
)
    chain, levels, ordinal_mappings, _ = fitresult
    Xnew = ordinal_encoder_transform(Xnew, ordinal_mappings)
    X = _f32(reformat(Xnew), 0)
    probs = vec(chain(X))
    return MLJModelInterface.UnivariateFinite(levels, probs; augment = true)
end

MLJModelInterface.metadata_model(
    NeuralNetworkBinaryClassifier,
    input_scitype = Union{AbstractMatrix{Continuous}, Table(Continuous, Finite)},
    target_scitype = AbstractVector{<:Finite{2}},
    load_path = "MLJFlux.NeuralNetworkBinaryClassifier",
)
