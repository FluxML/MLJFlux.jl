## EXPOSE OPTIMISERS TO MLJ (for eg, tuning)

# make the optimiser structs "transparent" so that their field values
# are exposed by calls to MLJ.params:
MLJModelInterface.istransparent(m::Optimisers.AbstractRule) = true


## GENERAL METHOD TO OPTIMIZE A CHAIN

struct Mover{R<:AbstractResource}
    resource::R
end

(::Mover{<:CPU1})(data) = Flux.cpu(data)
(::Mover{<:CUDALibs})(data) = Flux.gpu(data)

"""
    train_epoch(
        model,
        chain,
        optimiser,
        optimiser_state,
        X,
        y,
    ) -> updated_chain, updated_optimiser_state, training_loss

Update the parameters of a Flux `chain`, where:

- `model` is typically an `MLJFluxModel` instance, but could be any object such that
  `model.loss` is a Flux.jl loss function.

- the loss function `(yhat, y) -> loss(yhat, y)` is inferred from the
  `model`

- `X` and `y` are vectors of batches of the training data, as detailed
  in the [`MLJFlux.train`](@ref) document string.

"""
function train_epoch(
    model,
    chain,
    optimiser,
    optimiser_state,
    X,
    y,
    )

    loss = model.loss
    n_batches = length(y)
    training_loss = zero(Float32)

    for i in 1:n_batches
        batch_loss, gs = Flux.withgradient(chain) do m
            yhat = m(X[i])
            loss(yhat, y[i])
        end
        training_loss += batch_loss
        # The `do` syntax above means `gs` is a tuple of length one we need to unwrap to
        # get the actual gradient:
        ∇ = first(gs)
        optimiser_state, chain = Optimisers.update(optimiser_state, chain, ∇)
    end

    return chain, optimiser_state, training_loss / n_batches
end


"""
    train(
        model,
        chain,
        optimiser,
        optimiser_state,
        epochs,
        verbosity,
        X,
        y,
    ) -> (updated_chain, updated_optimiser_state, history)

Optimize a Flux model `chain`, where `(yhat, y) -> loss(yhat, y)` is the loss function
inferred from the `model`. Typically, `model` will be an `MLJFluxModel` instance, but it
could be any object such that `model.loss` is a Flux.jl loss function.

Here `chain` is a `Flux.Chain` object, or other Flux model such that
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

Both the `chain` and the data `(X, y)` must both live on a CPU or both
live on a GPU. This `train` method takes no responsibility for data
movement.

# Return value

Returns `(updated_chain, updated_optimiser_state, history)`, where `updated_chain` is a
trained version of `chain` and `history` is a vector of losses, including the
initial (no-train) loss.

"""
function train(
    model,
    chain,
    optimiser,
    optimiser_state,
    epochs,
    verbosity,
    X,
    y,
    )

    loss = model.loss

    # intitialize and start progress meter:
    meter = Progress(epochs + 1, dt=0, desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)

    # initiate history:
    n_batches = length(y)

    losses = (loss(chain(X[i]), y[i]) for i in 1:n_batches)
    history = [mean(losses),]

    for i in 1:epochs
        chain, optimiser_state, current_loss = train_epoch(
            model,
            chain,
            optimiser,
            optimiser_state,
            X,
            y,
        )
        verbosity < 2 ||
            @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    return chain, optimiser_state, history

end

# TODO: add callback functionality to above.



## HELPERS

"""
    gpu_isdead()

Returns `true` if `acceleration=CUDALibs()` option is unavailable, and
false otherwise.

"""
gpu_isdead() = Flux.gpu([1.0,]) isa Array

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
nrows(X::AbstractMatrix) = size(X, 1)

reformat(X) = reformat(X, scitype(X))

# ---------------------------------
# Reformatting matrices
reformat(X, ::Type{<:AbstractMatrix}) = X'

# ---------------------------------
# Reformatting tables
reformat(X, ::Type{<:Table}) = MLJModelInterface.matrix(X)'

# ---------------------------------
# Reformatting images

reformat(X, ::Type{<:GrayImage}) =
    reshape(Float32.(X), size(X)..., 1)

function reformat(X, ::Type{<:AbstractVector{<:GrayImage}})
    ret = zeros(Float32, size(first(X))..., 1, length(X))
    for idx = 1:size(ret, 4)
        ret[:, :, :, idx] .= reformat(X[idx])
    end
    return ret
end

function reformat(X, ::Type{<:ColorImage})
    ret = zeros(Float32, size(X)..., 3)
    for w = 1:size(X)[1]
        for h = 1:size(X)[2]
            ret[w, h, :] .= Float32.([X[w, h].r, X[w, h].g, X[w, h].b])
        end
    end
    return ret
end

function reformat(X, ::Type{<:AbstractVector{<:ColorImage}})
    ret = zeros(Float32, size(first(X))..., 3, length(X))
    for idx = 1:size(ret, 4)
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

reformat(y, ::Type{<:AbstractVector{<:Union{Continuous,Count}}}) =
    reshape(y, 1, length(y))
function reformat(y, ::Type{<:AbstractVector{<:Finite}})
    levels = y |> first |> MLJModelInterface.classes
    return Flux.onehotbatch(y, levels)
end

_get(Xmatrix::AbstractMatrix, b) = Xmatrix[:, b]
_get(y::AbstractVector, b) = y[b]

# each element in X is a single image of size (w, h, c)
_get(X::AbstractArray{<:Any,4}, b) = X[:, :, :, b]


"""
    collate(model, X, y)

**Private method**

Return the Flux-friendly data object required by `MLJFlux.train`, given
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
function collate(model::NeuralNetworkBinaryClassifier, X, y)
    row_batches = Base.Iterators.partition(1:nrows(y), model.batch_size)
    Xmatrix = reformat(X)
    yvec = (y .== classes(y)[2])' # convert to boolean
    return [_get(Xmatrix, b) for b in row_batches], [_get(yvec, b) for b in row_batches]
end
