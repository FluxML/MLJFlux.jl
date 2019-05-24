# using Revise
using Test
import MLJBase
import FluxMLJ
using LinearAlgebra
using CategoricalArrays
using Statistics
import Flux
import Random.seed!
seed!(123)

# test equality of optimisers:
@test Flux.Momentum() == Flux.Momentum()
@test Flux.Momentum(0.1) != Flux.Momentum(0.2)


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

builder = FluxMLJ.Linear(σ=identity)
model = FluxMLJ.NeuralNetworkRegressor(loss=mse, builder=builder)
fitresult, cache, report =
    MLJBase.fit(model, 1, MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))
@test mse(yhat, y[test]) <= 0.001

# univariate targets are ordinary vectors.

y_univariate = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5

uni_model = FluxMLJ.NeuralNetworkRegressor(loss=mse, builder=builder)
fitresult, cache, report = MLJBase.fit(model, 1, MLJBase.selectrows(X,train), y_univariate[train])

yhat_uni = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

@test mse(yhat_uni, y_univariate[test]) <= 0.001

## Classifier test:
## To Do: add test for loss function.

N = 200
X = MLJBase.table(randn(10N, 5))

# while multivariate targets are vectors of tuples:
ymatrix = hcat(1 .+ X.x1 - X.x2 .+ 1 .- 2X.x4 + X.x5)
train = 1:7N
test = (7N+1):10N
ymatrix = ymatrix ./ maximum(ymatrix)

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

y = CategoricalArray(get_labels.(ymatrix));

builder = FluxMLJ.Linear(σ=Flux.sigmoid)
model = FluxMLJ.NeuralNetworkClassifier(loss=Flux.crossentropy, builder=builder)
fitresult, cache, report =
    MLJBase.fit(model, 1, MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

@test size(yhat) == (600,)