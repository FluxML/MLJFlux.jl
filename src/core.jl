## EXPOSE OPTIMISERS TO MLJ (for eg, tuning)

# Here we make the optimiser structs "transparent" so that their
# field values are exposed by calls to MLJ.params

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
    end
end


## GENERAL METHOD TO OPTIMIZE A CHAIN

struct Mover{R<:AbstractResource}
    resource::R
end

(::Mover{<:CPU1})(data) = Flux.cpu(data)
(::Mover{<:CUDALibs})(data) = Flux.gpu(data)

"""
    train!(loss_func, parameters, optimiser, X, y)

A private method.

Update the parameters of a Flux chain with parameters `parameters`,
given a Flux-style "loss function" `loss(x, y)`. Here `X` and `y` are
vectors of batches of the training data, as detailed in the
[`MLJFlux.fit!`](@ref) document string.

"""
function train!(loss_func, parameters, optimiser, X, y)
    n_batches = length(y)
    training_loss = zero(Float32)
    for i in 1:n_batches
        gs = Flux.gradient(parameters) do
            batch_loss = loss_func(X[i], y[i])
            training_loss += batch_loss
            return batch_loss
        end
        Flux.update!(optimiser, parameters, gs)
    end
    return training_loss/n_batches
end


"""
    fit!(penalized_loss, chain, optimiser, epochs, verbosity, X, y)

A private method.

Optimize a Flux model `chain`, where `penalized_loss(xb, yb)` is the
penalized loss associated with a batch of training input features `xb`
and target observations `yb` (and generally depends on `chain`).

Here `chain` is a `Flux.Chain` object, or other "Flux model" such that
`Flux.params(chain)` returns the parameters to be optimized.

`X`, the vector of input batches and `y` the vector of target
batches. Specifically, it is expected that:

- `X` and `y` have type `Vector{<:Array{<:AbstractFloat}}`

- The shape of each element of `X` is `(n1, n2, ..., nk, batch_size)`
  where `(n1, n2, ..., nk)` is the shape of the inputs of `chain`

- The shape of each element of `y` is `(m1, m2, ..., mk, batch_size)`
  where `(m1, m2, ..., mk)` is the shape of the `chain` outputs (even
  if `batch_size == 1`).

- The vectors `X` and `y` have the same length, coinciding with the
  total number of training batches.

The [`MLJFlux.PenalizedLoss`](@ref) constructor is available for
defining an appropriate `penalized_loss` from an MLJFlux model and
chain (the model specifies the unpenalized Flux loss, such as `mse`,
and the regularization parameters).

Both the `chain` and the data `(X, y)` must both live on a CPU or both
live on a GPU. This `fit!` method takes no responsibility for data
movement.

### Return value

`(chain_trained, history)`, where `chain_trained` is a trained version
of `chain` and `history` is a vector of penalized losses - one initial
loss, and one loss per epoch.

"""
function  fit!(penalized_loss, chain, optimiser, epochs, verbosity, X, y)

    # intitialize and start progress meter:
    meter = Progress(epochs+1, dt=0, desc="Optimising neural net:",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)

    # initiate history:
    n_batches = length(y)

    training_loss = mean(penalized_loss(X[i], y[i]) for i in 1:n_batches)
    history = [training_loss,]

    for i in 1:epochs
        current_loss = train!(penalized_loss,
                              Flux.params(chain),
                              optimiser,
                              X,
                              y)
        verbosity < 2 ||
            @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    return chain, history

end

# TODO: add callback functionality to above.



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
# Reformatting vectors of length n into matrices of dimension n * 1
# This enables compatibility with Flux's BatchNorm. This is currently
# used only in `predict`. In the future, when MLJ's "data front end"
# is implemented, `tomat` and the `reformat` of continuous vectors
# that follows will be collapsed and there will be some
# simplification.

function tomat end
tomat(x::Matrix) = x
tomat(y::Vector) = reshape(y, size(y, 1), 1)

# ------------------------------------------------------------
# Reformatting vectors of "scalar" types

reformat(y, ::Type{<:AbstractVector{<:Continuous}}) = reshape(y, 1, length(y))
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
