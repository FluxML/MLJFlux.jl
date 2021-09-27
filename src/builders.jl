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
    Linear(; σ=Flux.relu, rng=Random.GLOBAL_RNG)

MLJFlux builder that constructs a fully connected two layer network
with activation function `σ`. The number of input and output nodes is
determined from the data. The bias and coefficients are initialized
using `Flux.glorot_uniform(rng)`. If `rng` is an integer, it is
instead used as the seed for a `MersenneTwister`.

"""
mutable struct Linear <: Builder
    σ
end
Linear(; σ=Flux.relu) = Linear(σ)
build(builder::Linear, rng, n::Integer, m::Integer) =
    Flux.Chain(Flux.Dense(n, m, builder.σ, init=Flux.glorot_uniform(rng)))

"""
    Short(; n_hidden=0, dropout=0.5, σ=Flux.sigmoid, rng=GLOBAL_RNG)

MLJFlux builder that constructs a full-connected three-layer network
using `n_hidden` nodes in the hidden layer and the specified `dropout`
(defaulting to 0.5). An activation function `σ` is applied between the
hidden and final layers. If `n_hidden=0` (the default) then `n_hidden`
is the geometric mean of the number of input and output nodes.  The
number of input and output nodes is determined from the data. 

The each layer is initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.

"""
mutable struct Short <: Builder
    n_hidden::Int     # if zero use geometric mean of input/output
    dropout::Float64
    σ
end
Short(; n_hidden=0, dropout=0.5, σ=Flux.sigmoid) = Short(n_hidden, dropout, σ)
function build(builder::Short, rng, n, m)
    n_hidden =
        builder.n_hidden == 0 ? round(Int, sqrt(n*m)) : builder.n_hidden
    init=Flux.glorot_uniform(rng)
    Flux.Chain(
        Flux.Dense(n, n_hidden, builder.σ, init=init),
        # TODO: fix next after https://github.com/FluxML/Flux.jl/issues/1617
        Flux.Dropout(builder.dropout),
        Flux.Dense(n_hidden, m, init=init))
end

"""
    MLP(; hidden=(10,), σ=Flux.sigmoid, rng=GLOBAL_RNG)

MLJFlux builder that constructs a Multi-layer perceptron network. The ith element of `hidden` represents the number of neurons in the ith hidden layer. An activation function `σ` is applied between each layer.

The each layer is initialized using `Flux.glorot_uniform(rng)`. If
`rng` is an integer, it is instead used as the seed for a
`MersenneTwister`.

"""
mutable struct MLP{N} <: MLJFlux.Builder
    hidden::NTuple{N, Int}  # count first and last layer
    σ
end
MLP(; hidden=(10,), σ=Flux.sigmoid) = MLP(hidden, σ)
function MLJFlux.build(mlp::MLP, rng, n_in, n_out)
    init=Flux.glorot_uniform(rng)
    hidden = []
    for i ∈ 2:length(mlp.hidden)
        push!(hidden, Flux.Dense(mlp.hidden[i-1], mlp.hidden[i], mlp.σ, init=init))
    end

    return Flux.Chain(
        Flux.Dense(n_in, mlp.hidden[1], mlp.σ, init=init),
        hidden...,
        Flux.Dense(mlp.hidden[end], n_out, init=init)
    )
end








struct GenericBuilder{F} <: Builder
    apply::F
end

"""
    @builder neural_net

Creates a builder for `neural_net`. The variables `rng`, `n_in`, `n_out` and
`n_channels` can be used to create builders for any random number generator `rng`,
input and output sizes `n_in` and `n_out` and number of input channels `n_channels`.

# Examples
```jldoctest
julia> import MLJFlux: @builder;

julia> nn = NeuralNetworkRegressor(builder = @builder(Chain(Dense(n_in, 64, relu),
                                                            Dense(64, 32, relu),
                                                            Dense(32, n_out))));

julia> conv_builder = @builder begin
           front = Chain(Conv((3, 3), n_channels => 16), Flux.flatten)
           d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
           Chain(front, Dense(d, n_out));
       end

julia> conv_nn = NeuralNetworkRegressor(builder = conv_builder);
```
"""
macro builder(ex)
    esc(quote
        MLJFlux.GenericBuilder((rng, n_in, n_out, n_channels) -> $ex)
    end)
end

build(b::GenericBuilder, rng, n_in, n_out, n_channels = 1) = b.apply(rng, n_in, n_out, n_channels)
