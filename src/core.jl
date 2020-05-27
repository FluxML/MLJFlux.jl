## EXPOSE OPTIMISERS TO MLJ (for eg, tuning)

# Here we: (i) Make the optimiser structs "transarent" so that their
# field values are exposed by calls to MLJ.params; and (ii) Overload
# `==` for optimisers, so that we can detect when their parameters
# remain unchanged on calls to MLJModelInterface.update methods.

# We define optimisers of to be `==` if: (i) They have identical type
# AND (ii) their defined field values are `==`. (Note that our `fit`
# methods will only use deep copies of optimisers specified as
# hyperparameters because some fields of `optimisers` carry "state"
# information which is mutated during chain updates.)

for opt in (:Descent, :Momentum, :Nesterov, :RMSProp, :ADAM, :AdaMax,
        :ADAGrad, :ADADelta, :AMSGrad, :NADAM, :Optimiser,
        :InvDecay, :ExpDecay, :WeightDecay)

    @eval begin

# TODO: Uncomment next line when
# https://github.com/alan-turing-institute/MLJModelInterface.jl/issues/28
# is resolved:

        # MLJModelInterface.istransparent(m::Flux.$opt) = true

        function ==(m1::Flux.$opt, m2::Flux.$opt)
            same_values = true
            for fld in fieldnames(Flux.$opt)
                same_values = same_values &&
                    getfield(m1, fld) == getfield(m2, fld)
            end
            return same_values
        end

    end

end


## GENERAL METHOD TO OPTIMIZE A CHAIN

"""
    fit!(chain,
         optimiser,
         loss,
         epochs,
         lambda,
         alpha,
         verbosity,
         data)

Optimize a Flux model `chain` using the regularization parameters
`lambda` (strength) and `alpha` (l2/l1 mix), where `loss(yhat, y) ` is
the supervised loss for instances (or vectors of instances) of the
target predictions `yhat` and target observations `y`.

Here `chain` is a `Flux.Chain` object, or other "Flux model" such that
`Flux.params(chain)` returns the parameters to be optimised.

The training `data` is a vector of tuples of the form `(X, y)` where:

- `X` and `y` have type `Array{<:AbstractFloat}`

- the shape of `X` is `(n1, n2, ..., nk, batch_size)` where `(n1, n2,
  ..., nk)` is the shape of the inputs of `chain`

- the shape of `y` is `(m1, m2, ..., mk, batch_size)` where `(m1, m2,
  ..., mk)` is the shape of the `chain` outputs.

The contribution to the objective function of a single input/output
instance `(X, y)` is

    loss(chain(X), y) + lambda*(model.alpha*l1) + (1 - model.alpha)*l2

where `l1 = sum(norm, params(chain)` and `l2 = sum(norm, params(chain))`.

"""
function  fit!(chain, optimiser, loss, epochs,
               lambda, alpha, verbosity, data)

    Flux.testmode!(chain, false)
    # intitialize and start progress meter:
    meter = Progress(epochs+1, dt=0, desc="Optimising neural net:",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)
    loss_func(x, y) = loss(chain(x), y)
    history = []
    prev_loss = Inf
    for i in 1:epochs
        # We're taking data in a Flux-fashion.
        Flux.train!(loss_func, Flux.params(chain), data, optimiser)
        current_loss =
            mean(loss_func(data[i][1], data[i][2]) for i=1:length(data))
        verbosity < 2 ||
            @info "Loss is $(round(current_loss.data; sigdigits=4))"
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
    Flux.testmode!(chain, true)         # to use in inference mode
    return chain, history

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

abstract type Builder <: MLJModelInterface.Model end

# baby example 1:
mutable struct Linear <: Builder
    σ
end
Linear(; σ=Flux.relu) = Linear(σ)
fit(builder::Linear, n::Integer, m::Integer) =
    Flux.Chain(Flux.Dense(n, m, builder.σ))

# baby example 2:
mutable struct Short <: Builder
    n_hidden::Int     # if zero use geometric mean of input/outpu
    dropout::Float64
    σ
end
Short(; n_hidden=0, dropout=0.5, σ=Flux.sigmoid) = Short(n_hidden, dropout, σ)
function fit(builder::Short, n, m)
    n_hidden =
        builder.n_hidden == 0 ? round(Int, sqrt(n*m)) : builder.n_hidden
    return Flux.Chain(Flux.Dense(n, n_hidden, builder.σ),
                      Flux.Dropout(builder.dropout),
                       Flux.Dense(n_hidden, m))
end


## HELPERS

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

function reformat(y, ::Type{<:AbstractVector{<:Multiclass}})
    levels = y |> first |> MLJModelInterface.classes
    return hcat([Flux.onehot(ele, levels) for ele in y]...,)
end

_get(Xmatrix::AbstractMatrix, b) = Xmatrix[:, b]
_get(y::AbstractVector, b) = y[b]

# each element in X is a single image of size (w, h, c)
function _get(Xmatrix::AbstractArray{T, 4}, b) where T
    return Xmatrix[:, :, :, b]
end

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
    return [(_get(Xmatrix, b), _get(ymatrix, b)) for b in row_batches]
end
