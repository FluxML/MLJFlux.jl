## BUILDING CHAINS A FROM HYPERPARAMETERS + INPUT/OUTPUT SHAPE

# We introduce chain builders as a way of exposing neural network
# hyperparameters (describing, architecture, dropout rates, etc) to
# the MLJ user. These parameters generally exclude the input/output
# dimensions/shapes, as the MLJ fit methods will determine these from
# the training data. A `Builder` object stores the parameters and an
# associated `fit` method builds a corresponding chain given the
# input/output dimensions/shape.

# Below n or (n1, n2) etc refers to network inputs, while m or (m1,
# m2) etc refers to outputs.

abstract type Builder <: MLJModelInterface.MLJType end

"""
    Linear(; σ=Flux.relu)

MLJFlux builder that constructs a fully connected two layer network
with activation function `σ`. The number of input and output nodes is
determined from the data.

"""
mutable struct Linear <: Builder
    σ
end
Linear(; σ=Flux.relu) = Linear(σ)
build(builder::Linear, n::Integer, m::Integer) =
    Flux.Chain(Flux.Dense(n, m, builder.σ))

"""
    Short(; n_hidden=0, dropout=0.5, σ=Flux.sigmoid)

MLJFlux builder that constructs a full-connected three-layer network
using `n_hidden` nodes in the hidden layer and the specified `dropout`
(defaulting to 0.5). An activation function `σ` is applied between the
hidden and final layers. If `n_hidden=0` (the default) then `n_hidden`
is the geometric mean of the number of input and output nodes.  The
number of input and output nodes is determined from the data.

"""
mutable struct Short <: Builder
    n_hidden::Int     # if zero use geometric mean of input/output
    dropout::Float64
    σ
end
Short(; n_hidden=0, dropout=0.5, σ=Flux.sigmoid) = Short(n_hidden, dropout, σ)
function build(builder::Short, n, m)
    n_hidden =
        builder.n_hidden == 0 ? round(Int, sqrt(n*m)) : builder.n_hidden
    return Flux.Chain(Flux.Dense(n, n_hidden, builder.σ),
                      Flux.Dropout(builder.dropout),
                       Flux.Dense(n_hidden, m))
end

