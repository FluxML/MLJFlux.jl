### Defining Custom Builders

Following is an example defining a new builder for creating a simple
fully-connected neural network with two hidden layers, with `n1` nodes
in the first hidden layer, and `n2` nodes in the second, for use in
any of the first three models in Table 1. The definition includes one
mutable struct and one method:

```julia
mutable struct MyBuilder <: MLJFlux.Builder
	n1 :: Int
	n2 :: Int
end

function MLJFlux.build(nn::MyBuilder, rng, n_in, n_out)
	init = Flux.glorot_uniform(rng)
        return Chain(
            Dense(n_in, nn.n1, init=init),
            Dense(nn.n1, nn.n2, init=init),
            Dense(nn.n2, n_out, init=init),
            )
end
```

Note here that `n_in` and `n_out` depend on the size of the data (see
[Table 1](@ref Models).

For a concrete image classification example, see [Using MLJ to classifiy the MNIST image
dataset](@ref).

More generally, defining a new builder means defining a new struct
sub-typing `MLJFlux.Builder` and defining a new `MLJFlux.build` method
with one of these signatures:

```julia
MLJFlux.build(builder::MyBuilder, rng, n_in, n_out)
MLJFlux.build(builder::MyBuilder, rng, n_in, n_out, n_channels) # for use with `ImageClassifier`
```

This method must return a `Flux.Chain` instance, `chain`, subject to the
following conditions:

- `chain(x)` must make sense:
  - for any `x <: Array{<:AbstractFloat, 2}` of size `(n_in,
    batch_size)` where `batch_size` is any integer (for use with one
    of the first three model types); or
  - for any `x <: Array{<:Float32, 4}` of size `(W, H, n_channels,
    batch_size)`, where `(W, H) = n_in`, `n_channels` is 1 or 3, and
    `batch_size` is any integer (for use with `ImageClassifier`)

- The object returned by `chain(x)` must be an `AbstractFloat` vector
  of length `n_out`.

Alternatively, use [`MLJFlux.@builder(neural_net)`](@ref) to automatically create a builder for
any valid Flux chain expression `neural_net`, where the symbols `n_in`, `n_out`,
`n_channels` and `rng` can appear literally, with the interpretations explained above. For
example,

```
builder = MLJFlux.@builder Chain(Dense(n_in, 128), Dense(128, n_out, tanh))
```
