# Note (1). See
# https://discourse.julialang.org/t/weight-regularisation-which-iterates-params-m-in-flux-mutating-arrays-is-not-supported/64314


""" Penalizer(λ, α)

Returns a callable object `penalizer` for evaluating regularization
penalties associated with some numerical array. Specifically,
`penalizer(A)` returns

   λ*(α*L1 + (1 - α)*L2),

where `L1` is the sum of absolute values of the elments of `A` and
`L2` is the sum of squares of those elements.
"""
struct Penalizer{T}
    lambda::T
    alpha::T
    function Penalizer(lambda, alpha)
        lambda == 0 && return new{Nothing}(nothing, nothing)
        T = promote_type(typeof.((lambda, alpha))...)
        return new{T}(lambda, alpha)
    end
end

(::Penalizer{Nothing})(::Any) = 0
function (p::Penalizer)(A)
    λ = p.lambda
    α = p.alpha
    # avoiding broadcasting; see Note (1) above
    L2 = sum(x^2 for x in A)
    L1 = sum(abs(x) for x in A)
    return  λ*(α*L1 + (1 - α)*L2)
end

"""
    PenalizedLoss(model, chain)

Returns a callable object `p`, for returning the penalized loss on
some batch of data `(x, y)`. Specifically, `p(x, y)` returns

   loss(chain(x), y) + sum(Penalizer(λ, α).(params(chain)))

where `loss = model.loss`, `α = model.alpha`, `λ = model.lambda`.

See also [`Penalizer`](@ref)

"""
struct PenalizedLoss{P}
    loss
    penalizer::P
    chain
    params
    function PenalizedLoss(model, chain)
        loss = model.loss
        penalizer = Penalizer(model.lambda, model.alpha)
        params = Flux.params(chain)
        return new{typeof(penalizer)}(loss, penalizer, chain, params)
    end
end
(p::PenalizedLoss{Penalizer{Nothing}})(x, y) = p.loss(p.chain(x), y)
(p::PenalizedLoss)(x, y) = p.loss(p.chain(x), y) +
    sum(p.penalizer(θ) for θ in p.params)
