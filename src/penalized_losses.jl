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
    L2 = sum(abs2, A)
    L1 = sum(abs,  A)
    return  λ*(α*L1 + (1 - α)*L2)
end



"""
    Penalty(model)

Returns a callable object `p`, for returning the regularization
penalty `p(w)` associated with some collection of parameters `w`. For
example, `w = params(chain)` where `chain` is some Flux
model. Here `model` is an MLJFlux model ("model" in the MLJ
sense, not the Flux sense). Specifically, `p(w)` returns

   sum(Penalizer(λ, α).w)

where `α = model.alpha`, `λ = model.lambda`.

See also [`Penalizer`](@ref)

"""
struct Penalty{P}
    penalizer::P
    function Penalty(model)
        penalizer = Penalizer(model.lambda, model.alpha)
        return new{typeof(penalizer)}(penalizer)
    end
end
(p::Penalty{Penalizer{Nothing}})(w) = 0
(p::Penalty)(w) = sum(p.penalizer(wt) for wt in w)
