module FluxMLJ

export NeuralNetworkRegressor, UnivariateNeuralNetworkRegressor
export NeuralNetworkClassifier, UnivariateNeuralNetworkClassifier

# import LossFunctions
import Flux
import MLJBase
import Base.==
using ProgressMeter


# CONSTANTS

# const Loss = LossFunctions.SupervisedLoss # owned by LearnBase


## HELPERS

nrows(X::AbstractMatrix) = size(X, 2)


## EXPOSE OPTIMISERS TO MLJ (for eg, tuning)

## Need MLJBase >=0.2.1 for this.

# Here we: (i) Make the optimiser structs "transarent" so that their
# field values are exposed by calls to MLJ.params (needed for tuning);
# and (ii) Overload == for optimisers, so that we can detect when
# their parameters remain unchanged on calls to MLJBase.update methods.

# We define optimisers of to be `==` if: (i) They have identical type
# AND (ii) their defined field values are `==`:

for opt in (:Descent, :Momentum, :Nesterov, :RMSProp, :ADAM, :AdaMax,
        :ADAGrad, :ADADelta, :AMSGrad, :NADAM, :Optimiser,
        :InvDecay, :ExpDecay, :WeightDecay)

    @eval begin
        
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
        current_loss = sum(loss_func(data[i][1], data[i][2]) for i=1:length(data))
        verbosity < 2 || println("Loss is $(current_loss.data)")
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
    return chain, map(x->x.data, history)

end


# TODO: add automatic stopping and callback functionality to above.


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

abstract type Builder <: MLJBase.Model end

# baby example 1:
mutable struct Linear <: Builder
    σ
end
Linear(; σ=Flux.sigmoid) = Linear(σ)
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
# to optimise the neural network it builds (after seeing the data). It
# additionally specifies a training loss function and loss penalty
# regularization parameters.


## NEURAL NETWORK REGRESSOR

mutable struct NeuralNetworkRegressor{B<:Builder,O,L} <: MLJBase.Deterministic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end
NeuralNetworkRegressor(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.mse
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining=false) where {B,O,L} =
                  NeuralNetworkRegressor{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)

input_is_multivariate(::Type{<:NeuralNetworkRegressor}) = true
input_scitype_union(::Type{<:NeuralNetworkRegressor}) = MLJBase.Continuous 
target_scitype_union(::Type{<:NeuralNetworkRegressor}) =
    Union{MLJBase.Continuous,NTuple{<:MLJBase.Continuous}}
    
function collate(X, y, batch_size, regressor=true)
    Xmatrix = MLJBase.matrix(X)'  # TODO: later MLJBase.matrix(X_,
                                   # transpose=true)
    ymatrix = reduce(hcat, [[tup...] for tup in y])
    row_batches = Base.Iterators.partition(1:length(y), batch_size)
    if regressor
        if y isa AbstractVector{<:Tuple}
            return [(Xmatrix[:, b], ymatrix[:, b]) for b in row_batches]
        else
            return [(Xmatrix[:, b], ymatrix[b]) for b in row_batches]
        end
    else
        a_target_element = first(y)
        levels = MLJBase.classes(a_target_element)
        ymatrix = hcat([Flux.onehot(ele, levels) for ele in y]...,)
        return [(Xmatrix[:, b], ymatrix[:, b]) for b in row_batches]
    end
end

function MLJBase.fit(model::NeuralNetworkRegressor,
                     verbosity::Int,
                     X, y)

    data = collate(X, y, model.batch_size)

    target_is_multivariate = y isa AbstractVector{<:Tuple}
    
    n = MLJBase.schema(X).names |> length
    m = length(y[1])

    chain = fit(model.builder, n, m)

    # fit!(chain,...) mutates optimisers!!
    # MLJ does not allow fit to mutate models. So:
    optimiser = deepcopy(model.optimiser)
    
    chain, history = fit!(chain, optimiser, model.loss, model.n, model.batch_size,
         model.lambda, model.alpha, verbosity, data)

    cache = (deepcopy(model), data, history) 
    fitresult = (chain, target_is_multivariate)
    report = (training_losses=history,)
    
    return fitresult, cache, report

end

# reformatting for a single prediction, according to whether target is
# multivariate or not:
reformat(ypred, ::Val{true}) = Tuple(ypred.data)
reformat(ypred, ::Val{false}) = first(ypred.data) 

# for multivariate targets:            
function MLJBase.predict(model, fitresult, Xnew_)
    chain = fitresult[1]
    ismulti = fitresult[2]
    Xnew = MLJBase.matrix(Xnew_)'
    return [reformat(chain(Xnew[:,i]), Val(ismulti)) for i in 1:size(Xnew, 2)]
end

function MLJBase.update(model::NeuralNetworkRegressor, verbosity::Int, old_fitresult, old_cache, X, y)

    old_model, data, old_history = old_cache
    old_chain, target_is_multivariate = old_fitresult

    keep_chain =  model.n >= old_model.n &&
        model.loss == old_model.loss &&
        model.batch_size == old_model.batch_size &&
        model.lambda == old_model.lambda &&
        model.alpha == old_model.alpha &&
        model.builder == old_model.builder &&
        (!model.optimiser_changes_trigger_retraining ||
         model.optimiser == old_model.optimiser)
    
    if keep_chain
        chain = old_chain
        epochs = model.n - old_model.n
    else
        n = MLJBase.schema(X).names |> length
        m = length(y[1])
        chain = fit(model.builder, n, m)
        epochs = model.n
    end
    
    # fit!(chain,...) mutates optimisers!!
    # MLJ does not allow fit to mutate models. So:
    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss, epochs,
                                model.batch_size, model.lambda, model.alpha,
                                verbosity, data)
    if keep_chain
        history = vcat(old_history, history)
    end
    fitresult = (chain, target_is_multivariate)
    cache = (deepcopy(model), data, history)
    report = (training_losses=history,)
    
    return fitresult, cache, report

end


## Classifier:

mutable struct NeuralNetworkClassifier{B<:Builder,O,L} <: MLJBase.Probabilistic
    builder::B
    optimiser::O    # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L         # can be called as in `loss(yhat, y)`
    n::Int          # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    optimiser_changes_trigger_retraining::Bool
end

NeuralNetworkClassifier(; builder::B   = Linear()
              , optimiser::O = Flux.Optimise.ADAM()
              , loss::L      = Flux.crossentropy
              , n            = 10
              , batch_size   = 1
              , lambda       = 0
              , alpha        = 0
              , optimiser_changes_trigger_retraining = false) where {B,O,L} =
                  NeuralNetworkClassifier{B,O,L}(builder
                                       , optimiser
                                       , loss
                                       , n
                                       , batch_size
                                       , lambda
                                       , alpha
                                       , optimiser_changes_trigger_retraining)

input_is_multivariate(::Type{<:NeuralNetworkClassifier}) = true
input_scitype_union(::Type{<:NeuralNetworkClassifier}) = MLJBase.Continuous 
target_scitype_union(::Type{<:NeuralNetworkClassifier}) =MLJBase.Multiclass

function MLJBase.fit(model::NeuralNetworkClassifier, verbosity::Int,
                        X_, y_)
    
    target_is_multivariate = y_ isa AbstractVector{<:Tuple}

    a_target_element = first(y_)
    levels = MLJBase.classes(a_target_element)
    m = length(levels)

    data = collate(X_, y_, model.batch_size, false)



    n = MLJBase.schema(X_).names |> length
    optimiser = deepcopy(model.optimiser)

    chain = fit(model.builder, n, m)

    chain, history = fit!(chain, model.optimiser, model.loss, model.n, model.batch_size,
    model.lambda, model.alpha, verbosity, data)

    cache = deepcopy(model), data, history # track number of epochs trained for update method
    fitresult = (chain, target_is_multivariate)

    report = (training_losses=history, epochs=model.n)

    return fitresult, cache, report
end

function MLJBase.predict(model::NeuralNetworkClassifier, fitresult, Xnew_)
    chain = fitresult[1]
    ismulti = fitresult[2]
    Xnew = MLJBase.matrix(Xnew_)'
    return [chain(Xnew[:,i]).data for i in 1:size(Xnew, 2)]

end

function MLJBase.update(model::NeuralNetworkClassifier, verbosity::Int, old_fitresult, old_cache, X, y)

    old_model, data, old_history = old_cache
    old_chain, target_is_multivariate = old_fitresult

    keep_chain =  model.n >= old_model.n &&
        model.loss == old_model.loss &&
        model.batch_size == old_model.batch_size &&
        model.lambda == old_model.lambda &&
        model.alpha == old_model.alpha &&
        model.builder == old_model.builder &&
        (!model.optimiser_changes_trigger_retraining ||
         model.optimiser == old_model.optimiser)
    
    if keep_chain
        chain = old_chain
        epochs = model.n - old_model.n
    else
        n = MLJBase.schema(X).names |> length
        m = length(y[1])
        chain = fit(model.builder, n, m)
        epochs = model.n
    end

    optimiser = deepcopy(model.optimiser)

    chain, history = fit!(chain, optimiser, model.loss, epochs,
                                model.batch_size, model.lambda, model.alpha,
                                verbosity, data)
    if keep_chain
        history = vcat(old_history, history)
    end
    fitresult = (chain, target_is_multivariate)
    cache = (deepcopy(model), data, history)
    report = (training_losses=history,)
    
    return fitresult, cache, report

end



end






    




    
              
              
