# if `b` is a builder, then `b(model, rng, shape...)` is called to make a
# new chain, where `shape` is the return value of this method:
"""
    shape(model::NeuralNetworkClassifier, X, y)

A private method that returns the shape of the input and output of the model for given data `X` and `y`.
"""
function MLJFlux.shape(model::NeuralNetworkClassifier, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    levels = MLJModelInterface.classes(y[1])
    n_output = length(levels)
    n_input = Tables.schema(X).names |> length
    return (n_input, n_output)
end

# builds the end-to-end Flux chain needed, given the `model` and `shape`:
MLJFlux.build(model::Union{NeuralNetworkClassifier, NeuralNetworkBinaryClassifier}, rng, shape) =
    Flux.Chain(build(model.builder, rng, shape...),
        model.finaliser)

# returns the model `fitresult` (see "Adding Models for General Use"
# section of the MLJ manual) which must always have the form `(chain,
# metadata)`, where `metadata` is anything extra needed by `predict`:
MLJFlux.fitresult(model::Union{NeuralNetworkClassifier, NeuralNetworkBinaryClassifier}, chain, y) =
    (chain, MLJModelInterface.classes(y[1]))

function MLJModelInterface.predict(model::NeuralNetworkClassifier,
    fitresult,
    Xnew)
    chain, levels = fitresult
    X = reformat(Xnew)
    probs = vcat([chain(tomat(X[:, i]))' for i in 1:size(X, 2)]...)
    return MLJModelInterface.UnivariateFinite(levels, probs)
end

MLJModelInterface.metadata_model(NeuralNetworkClassifier,
    input=Union{AbstractMatrix{Continuous},Table(Continuous)},
    target=AbstractVector{<:Finite},
    path="MLJFlux.NeuralNetworkClassifier")

#### Binary Classifier

function MLJFlux.shape(model::NeuralNetworkBinaryClassifier, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    n_input = Tables.schema(X).names |> length
    return (n_input, 1) # n_output is always 1 for a binary classifier
end

function MLJModelInterface.predict(model::NeuralNetworkBinaryClassifier,
    fitresult,
    Xnew)
    chain, levels = fitresult
    X = reformat(Xnew)
    probs = vec(chain(X))
    return MLJModelInterface.UnivariateFinite(levels, probs; augment = true)
end

MLJModelInterface.metadata_model(NeuralNetworkBinaryClassifier,
    input=Union{AbstractMatrix{Continuous},Table(Continuous)},
    target=AbstractVector{<:Finite{2}},
    path="MLJFlux.NeuralNetworkBinaryClassifier")
