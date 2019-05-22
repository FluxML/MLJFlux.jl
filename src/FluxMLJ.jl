module FluxMLJ

export NeuralNetworkRegressor, UnivariateNeuralNetworkRegressor
export NeuralNetworkClassifier, UnivariateNeuralNetworkClassifier

import Flux
import MLJBase
import LossFunctions
import Base.==
using ProgressMeter


# CONSTANTS

const Loss = LossFunctions.SupervisedLoss # owned by LearnBase


## HELPERS

nrows(X::AbstractMatrix) = size(X, 2)


## EXPOSE OPTIMISERS TO MLJ (for eg, tuning)

## Need MLJBase >=0.2.1 for this.

# Here we: (i) Make the optimser structs "transarent" so that their
# field values are exposed by calls to MLJ.params (needed for tuning);
# and (ii) Overload == for optimisers, so that we can detect when
# their parameters remain unchanged on calls to MLJBase.update methods.

# We define optimisers of to be `==` if: (i) They have identical type
# AND (ii) their defined field values are `==`:

for opt in (:Descent, :Momentum, :Nesterov, :RMSProp, :ADAM, :AdaMax,
        :ADAGrad, :ADADelta, :AMSGrad, :NADAM, :Optimiser,
        :InvDecay, :ExpDecay, :WeightDecay)

    @eval begin
        @show Flux.$opt
        
        MLJBase.istransparent(m::Flux.$opt) = true

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
        fit!(chain, optimiser, loss, epochs, batch_size, lambda, alpha, verbosity, data)

Optimize a Flux model `chain` using the regularization parameters
`lambda` (strength) and `alpha` (l2/l1 mix), where `loss(yhat, y) ` is
the supervised loss for instances of the target `yhat` and `y`.

Here `chain` is a `Flux.Chain` object, or other "Flux model" such that
`Flux.params(chain)` returns the parameters to be optimised.

The training `data` is a vector of tuples of the form `(X, y)` where:

- `X` and `y` have type `Array{<:AbstractFloat}` 

- the shape of `X` is `(n1, n2, ..., nk, batch_size)` where `(n1, n2,
  ..., nk)` is the shape of the inputs of `chain`

- the shape of `y` is `(m1, m2, ..., mk, batch_size)` where `(m1, m2,
  ..., mk)` is the shape of the `chain` outputs.

The contribution to the optimised objective function of a single
input/output instance `(X, y)` is

    loss(chain(X), y) + lambda*(model.alpha*l1 + (1 - model.alpha)*l2

where `l1 = sum(norm, params(chain)` and `l2 = sum(norm,
params(chain))`. 

"""
function  fit!(chain, optimiser, loss, epochs, batch_size,
               lambda, alpha, verbosity, data; min_loss = 0.5)

    # intitialize and start progress meter:
    meter = Progress(epochs+1, dt=0, desc="",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity < 1 || next!(meter)
    loss_func(x, y) = loss(chain(x), y)
    history = []
    for i in 1:epochs
        
        Flux.train!(loss_func, Flux.params(chain), data, optimiser)  # We're taking data in a Flux-fashion.
        current_loss = sum(loss_func(data[i][1], data[i][2]) for i=1:length(data))
        println("Loss is $current_loss")
        push!(history, current_loss)
        if (current_loss < min_loss)
            @info "Early stopping because we've reached desired accuracy"
            break
        end
        verbosity < 1 || next!(meter)
        
    end

    return chain, history

end


# TODO: add automatic stopping and callback functionality to above.


## BUILDING CHAINS A FROM HYPERPARAMETERs + INPUT/OUTPUT SHAPE 

# We introduce chain builders as a way of exposing neural network
# hyperparameters (describing, eg, architecture) to the MLJ
# user. These parameters generally exclude the input/output
# dimensions/shapes, as the MLJ fit methods will determine these from
# the training data. A `Builder` object stores the parameters
# and an associated `fit` method builds a corresponding chain given
# the input/output dimensions/shape.

# Below n or (n1, n2) etc refers to network inputs, while m or (m1,
# m2) etc refers to outputs.

abstract type Builder <: MLJBase.Model end

# baby example 1:
mutable struct Linear <: Builder
    σ
end
Linear(; σ=identity) = Linear(σ)
fit(builder::Linear, n::Integer, m::Integer) = Flux.Dense(n, m, builder.σ)

# baby example 2:
mutable struct Short <: Builder
    n_hidden::Int     # if zero use geometric mean of input/outpu 
    dropout::Float64
    σ
end
Short(; n_hidden=0, dropout=0.5, σ=tanh) = Short(n_hidden, dropout, σ)
function fit(builder::Short, n, m) 
    n_hidden =
        builder.n_hidden == 0 ? round(Int, sqrt(n*m)) : builder.n_hidden
    return Flux.Chain(Flux.Dense(n, n_hidden, builder.σ),
                      Flux.Dropout(builder.dropout),
                       Flux.Dense(n_hidden, m))
end


##############
# MLJ MODELS #
##############

# An MLJ model wraps a neural network builder with instructions on how
# to optimise the neural network it builds (after seeing the data).


## NEURAL NETWORK REGRESSOR

mutable struct NeuralNetworkRegressor{B<:Builder,O,L} <: MLJBase.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
end
NeuralNetworkRegressor(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = LossFunctions.L2DistLoss()
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0) where {B,O,L} =
                  NeuralNetworkRegressor{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha)

input_is_multivariate(::Type{<:NeuralNetworkRegressor}) = true
input_scitype_union(::Type{<:NeuralNetworkRegressor}) = Continuous 
target_scitype_union(::Type{<:NeuralNetworkRegressor}) =
    Union{Continuous,NTuple{<:Continuous}}
    
function MLJBase.fit(model::NeuralNetworkRegressor,
                     verbosity::Int,
                     X_, y_)
   
    target_is_multivariate = y_ isa AbstractVector{<:Tuple}

    # assemble as required by fit(chain,...) above:
    Xmatrix = MLJBase.matrix(X_)'  # TODO: later MLJBase.matrix(X_,
                                   # transpose=true)
    ymatrix = reduce(hcat, [[tup...] for tup in y_])
    row_batches = Base.Iterators.partition(1:length(y_), model.batch_size)
    if target_is_multivariate
        data = [(Xmatrix[:, b], ymatrix[:, b]) for b in row_batches]
    else
        data = [(Xmatrix[:, b], ymatrix[b]) for b in row_batches]
    end

    n = size(Xmatrix, 1)
    m = size(ymatrix, 1)

    chain = fit(model.builder, n, m)

    chain, report = fit!(chain, model.optimiser, model.loss, model.n, model.batch_size,
         model.lambda, model.alpha, verbosity, data)

    cache = model.n # track number of epochs trained for update method
    fitresult = (chain, target_is_multivariate)

    return fitresult, cache, report

end

# reformatting for a single prediction, according to whether target is
# multivariate or not:
reformat(ypred, ::Val{true}) = Tuple(ypred)
reformat(ypred, ::Val{false}) = first(ypred) 
# TODO: also strip of "tracked"

# for multivariate targets:            
function MLJBase.predict(model, fitresult, Xnew_)
    chain = fitresult[1]
    ismulti = fitresult[2]
    Xnew = MLJBase.matrix(Xnew_)'
    return [reformat(chain(Xnew[:,i]), Val(ismulti)) for i in 1:size(Xnew, 2)]
end

end






    




    
              
              
