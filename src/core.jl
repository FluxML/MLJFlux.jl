## EXPOSE OPTIMISERS TO MLJ (for eg, tuning)

# Here we: (i) Make the optimiser structs "transparent" so that their
# field values are exposed by calls to MLJ.params; and (ii) Overload
# `==` for optimisers, so that we can detect when their parameters
# remain unchanged on calls to MLJModelInterface.update methods.

# We define optimisers of to be `==` if: (i) They have identical type
# AND (ii) their defined field values are `==`. (Note that our `fit`
# methods will only use deep copies of optimisers specified as
# hyperparameters because some fields of `optimisers` carry "state"
# information which is mutated during chain updates.)

for opt in (:Descent,
            :Momentum,
            :Nesterov,
            :RMSProp,
            :ADAM,
            :RADAM,
            :AdaMax,
            :OADAM,
            :ADAGrad,
            :ADADelta,
            :AMSGrad,
            :NADAM,
            :AdaBelief,
            :Optimiser,
            :InvDecay, :ExpDecay, :WeightDecay,
            :ClipValue,
            :ClipNorm) # last updated: Flux.jl 0.12.3

    @eval begin
        MLJModelInterface.istransparent(m::Flux.$opt) = true
        ==(m1::Flux.$opt, m2::Flux.$opt) =
            MLJModelInterface._equal_to_depth_one(m1, m2)
    end
end


## GENERAL METHOD TO OPTIMIZE A CHAIN

struct Mover{R<:AbstractResource}
    resource::R
end

(::Mover{<:CPU1})(data) = Flux.cpu(data)
(::Mover{<:CUDALibs})(data) = Flux.gpu(data)

"""
Custom training loop. Here, `loss_func` is the objective
function to optimise, `parameters` are the model parameters,
`optimiser` is the optimizer to be used, `X` (input features)is a
vector of arrays where the last dimension is the batch size. `y`
is the target observation vector.
"""
function train!(loss_func, parameters, optimiser, X, y)
    for i=1:length(X)
        gs = Flux.gradient(parameters) do
            training_loss = loss_func(X[i], y[i])
            return training_loss
        end
        Flux.update!(optimiser, parameters, gs)
    end
end


"""
    fit!(chain,
         optimiser,
         loss,
         epochs,
         lambda,
         alpha,
         verbosity,
         acceleration,
         X,
         y)

Optimize a Flux model `chain` using the regularization parameters
`lambda` (strength) and `alpha` (l2/l1 mix), where `loss(yhat, y) ` is
the supervised loss for instances (or vectors of instances) of the
target predictions `yhat` and target observations `y`.

Here `chain` is a `Flux.Chain` object, or other "Flux model" such that
`Flux.params(chain)` returns the parameters to be optimised.

The `X` argument is the training features and `y` argument is the
target:

- `X` and `y` have type `Array{<:AbstractFloat}`

- the shape of `X` is `(n1, n2, ..., nk, batch_size)` where `(n1, n2,
  ..., nk)` is the shape of the inputs of `chain`

- the shape of `y` is `(m1, m2, ..., mk, batch_size)` where `(m1, m2,
  ..., mk)` is the shape of the `chain` outputs.

The contribution to the objective function of a single input/output
instance `(X, y)` is

    loss(chain(X), y) + lambda*(model.alpha*l1) + (1 - model.alpha)*l2

where `l1 = sum(norm, params(chain)` and `l2 = sum(norm, params(chain))`.

One must have `acceleration isa CPU1` or `acceleration isa CUDALibs`
(for running on a GPU) where `CPU1` and `CUDALibs` are types defined
in `ComputationalResources.jl`.

### Return value

`(chain_trained, history)`, where `chain_trained` is a trained version
of `chain` (possibly moved to a gpu) and `history` is a vector of
losses - one intial loss, and one loss per epoch. The method may
mutate the argument `chain`, depending on cpu <-> gpu movements.

"""
function  fit!(chain, optimiser, loss, epochs,
               lambda, alpha, verbosity, acceleration, X, y)

    # intitialize and start progress meter:
    meter = Progress(epochs+1, dt=0, desc="Optimising neural net:",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)

    move = Mover(acceleration)
    X = move(X)
    y = move(y)
    chain = move(chain)

    loss_func(x, y) = loss(chain(x), y)

    # initiate history:
    prev_loss = mean(loss_func(X[i], y[i]) for i=1:length(X))
    history = [prev_loss,]

    for i in 1:epochs
        # We're taking data in a Flux-fashion.
#        @show i rand()
        train!(loss_func, Flux.params(chain), optimiser, X, y)
        current_loss =
            mean(loss_func(X[i], y[i]) for i=1:length(X))
        verbosity < 2 ||
            @info "Loss is $(round(current_loss; sigdigits=4))"
        push!(history, current_loss)

        # Early stopping is to be externally controlled.
        # So @ablaom has commented next 5 lines :
        # if current_loss == prev_loss
        #     @info "Model has reached maximum possible accuracy."*
        #     "More training won't increase accuracy"
        #     break
        # end

        prev_loss = current_loss
        verbosity != 1 || next!(meter)

    end

    return Flux.cpu(chain), history

end

# TODO: add callback functionality to above.


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


## HELPERS

"""
    gpu_isdead()

Returns `true` if `acceleration=CUDALibs()` option is unavailable, and
false otherwise.

"""
gpu_isdead() = Flux.gpu([1.0, ]) isa Array

"""
    nrows(X)

Find the number of rows of `X`, where `X` is an `AbstractVector or
Tables.jl table.
"""
function nrows(X)
    Tables.istable(X) || throw(ArgumentError)
    Tables.columnaccess(X) || return length(collect(X))
    # if has columnaccess
    cols = Tables.columntable(X)
    !isempty(cols) || return 0
    return length(cols[1])
end
nrows(y::AbstractVector) = length(y)

reformat(X) = reformat(X, scitype(X))

# ---------------------------------
# Reformatting tables

reformat(X, ::Type{<:Table}) = MLJModelInterface.matrix(X)'

# ---------------------------------
# Reformatting images

reformat(X, ::Type{<:GrayImage}) =
    reshape(Float32.(X), size(X)..., 1)

function reformat(X, ::Type{<:AbstractVector{<:GrayImage}})
    ret = zeros(Float32, size(first(X))..., 1, length(X))
    for idx=1:size(ret, 4)
        ret[:, :, :, idx] .= reformat(X[idx])
    end
    return ret
end

function reformat(X, ::Type{<:ColorImage})
    ret = zeros(Float32, size(X)... , 3)
    for w = 1:size(X)[1]
        for h = 1:size(X)[2]
            ret[w, h, :] .= Float32.([X[w, h].r, X[w, h].g, X[w, h].b])
        end
    end
    return ret
end

function reformat(X, ::Type{<:AbstractVector{<:ColorImage}})
    ret = zeros(Float32, size(first(X))..., 3, length(X))
    for idx=1:size(ret, 4)
        ret[:, :, :, idx] .= reformat(X[idx])
    end
    return ret
end

# ------------------------------------------------------------
# Reformatting vectors of "scalar" types

reformat(y, ::Type{<:AbstractVector{<:Continuous}}) = y
function reformat(y, ::Type{<:AbstractVector{<:Finite}})
    levels = y |> first |> MLJModelInterface.classes
    return hcat([Flux.onehot(ele, levels) for ele in y]...,)
end

function reformat(y, ::Type{<:AbstractVector{<:Count}})
    levels = y |> first |> MLJModelInterface.classes
    return hcat([Flux.onehot(ele, levels) for ele in y]...,)
end

_get(Xmatrix::AbstractMatrix, b) = Xmatrix[:, b]
_get(y::AbstractVector, b) = y[b]

# each element in X is a single image of size (w, h, c)
_get(X::AbstractArray{<:Any, 4}, b) = X[:, :, :, b]


"""
    collate(model, X, y)

Return the Flux-friendly data object required by `MLJFlux.fit!`, given
input `X` and target `y` in the form required by
`MLJModelInterface.input_scitype(X)` and
`MLJModelInterface.target_scitype(y)`. (The batch size used is given
by `model.batch_size`.)

"""
function collate(model, X, y)
    row_batches = Base.Iterators.partition(1:nrows(y), model.batch_size)
    Xmatrix = reformat(X)
    ymatrix = reformat(y)
    return [_get(Xmatrix, b) for b in row_batches], [_get(ymatrix, b) for b in row_batches]
end
