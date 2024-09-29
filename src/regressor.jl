# # NEURAL NETWORK REGRESSOR

"""
    shape(model::NeuralNetworkRegressor, X, y)

A private method that returns the shape of the input and output of the model for given data `X` and `y`.
"""
function shape(model::NeuralNetworkRegressor, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    n_input = Tables.schema(X).names |> length
    n_ouput = 1
    return (n_input, 1)
end
is_embedding_enabled(::NeuralNetworkRegressor) = true


build(model::NeuralNetworkRegressor, rng, shape) =
    build(model.builder, rng, shape...)

fitresult(model::NeuralNetworkRegressor, chain, y, ordinal_mappings=nothing, embedding_matrices=nothing) =
    (chain, nothing, ordinal_mappings, embedding_matrices)



function MLJModelInterface.predict(model::NeuralNetworkRegressor,
    fitresult,
    Xnew)
    chain, ordinal_mappings = fitresult[1], fitresult[3]
    Xnew = ordinal_encoder_transform(Xnew, ordinal_mappings)
    Xnew_ = reformat(Xnew)
    return [chain(values.(tomat(Xnew_[:, i])))[1]
            for i in 1:size(Xnew_, 2)]
end

MLJModelInterface.metadata_model(NeuralNetworkRegressor,
    input = Union{AbstractMatrix{Continuous}, Table(Continuous,Finite)},
    target = AbstractVector{<:Continuous},
    path = "MLJFlux.NeuralNetworkRegressor")


# # MULTITARGET NEURAL NETWORK REGRESSOR

ncols(X::AbstractMatrix) = size(X, 2)
ncols(X) = Tables.columns(X) |> Tables.columnnames |> length

"""
    shape(model::MultitargetNeuralNetworkRegressor, X, y)

A private method that returns the shape of the input and output of the model for given
data `X` and `y`.
"""
shape(model::MultitargetNeuralNetworkRegressor, X, y) = (ncols(X), ncols(y))
is_embedding_enabled(::MultitargetNeuralNetworkRegressor) = true

build(model::MultitargetNeuralNetworkRegressor, rng, shape) =
    build(model.builder, rng, shape...)

function fitresult(
    model::MultitargetNeuralNetworkRegressor,
    chain,
    y,
    ordinal_mappings=nothing,
    embedding_matrices=nothing,
)
    if y isa Matrix
        target_column_names = nothing
    else
        target_column_names = Tables.schema(y).names
    end
    return (chain, target_column_names, ordinal_mappings, embedding_matrices)
end

function MLJModelInterface.predict(model::MultitargetNeuralNetworkRegressor,
    fitresult, Xnew)
    chain, target_column_names, ordinal_mappings, _ = fitresult
    Xnew = ordinal_encoder_transform(Xnew, ordinal_mappings)
    X = reformat(Xnew)
    ypred = [chain(values.(tomat(X[:, i])))
             for i in 1:size(X, 2)]
    output =
        isnothing(target_column_names) ? permutedims(reduce(hcat, ypred)) :
        MLJModelInterface.table(reduce(hcat, ypred)', names = target_column_names)
    return output
end

MLJModelInterface.metadata_model(MultitargetNeuralNetworkRegressor,
    input = Union{AbstractMatrix{Continuous}, Table(Continuous,Finite)},
    target = Union{AbstractMatrix{Continuous}, Table(Continuous)},
    path = "MLJFlux.MultitargetNeuralNetworkRegressor")
