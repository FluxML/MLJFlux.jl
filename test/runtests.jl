# using Revise
using Test
import MLJBase
import MLJFlux
using LinearAlgebra
using CategoricalArrays
using Statistics
using StatsBase
import Flux
import Random.seed!
seed!(123)

# test equality of optimisers:
@test Flux.Momentum() == Flux.Momentum()
@test Flux.Momentum(0.1) != Flux.Momentum(0.2)


## NEURAL NETWORK REGRESSOR

# in MLJ multivariate inputs are tables:
N = 200
X = MLJBase.table(randn(10N, 5))

# while multivariate targets are vectors of tuples:
ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5)
y = [Tuple(ymatrix[i,:]) for i in 1:size(ymatrix, 1)]

train = 1:7N
test = (7N+1):10N

se(yhat, y) = sum((yhat .- y).^2)
mse(yhat, y) = mean(broadcast(se, yhat, y))

builder = MLJFlux.Linear(σ=identity)
model = MLJFlux.NeuralNetworkRegressor(loss=mse, builder=builder)

fitresult, cache, report =
    MLJBase.fit(model, 1, MLJBase.selectrows(X,train), y[train])
model.n = 15
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))
@test mse(yhat, y[test]) <= 0.001

# univariate targets are ordinary vectors:
y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5

uni_model = MLJFlux.NeuralNetworkRegressor(loss=mse, builder=builder)

fitresult, cache, report =
    MLJBase.fit(uni_model, 1, MLJBase.selectrows(X,train), y[train])
model.n = 15
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(uni_model, fitresult, MLJBase.selectrows(X, test))

@test mse(yhat, y[test]) <= 0.001


## NEURAL NETWORK CLASSIFIER

## To Do: add test for loss function.

N = 100
X = MLJBase.table(randn(10N, 5))

yvector = 1 .+ X.x1 - X.x2 .+ 1 .- 2X.x4 + X.x5
train = 1:7N
test = (7N+1):10N
yvector = yvector ./ maximum(yvector)

l = ["1", "2", "3", "4"]

function get_labels(ele)
    if ele >0.5
        return "1"
    elseif ele > 0.0
        return "2"
    elseif ele > -0.5
        return "3"
    else
        return "4"
    end
end

y = CategoricalArray(get_labels.(yvector));

builder = MLJFlux.Linear(σ=Flux.sigmoid)
model = MLJFlux.NeuralNetworkClassifier(loss=Flux.crossentropy,
                                        builder=builder)
fitresult, cache, report =
    MLJBase.fit(model, 2, MLJBase.selectrows(X,train), y[train])
model.n = 15
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

misclassification_rate = sum(mode.(yhat) .!= y[test])/length(test)
@test misclassification_rate < 0.3

include("embeddings.jl")
