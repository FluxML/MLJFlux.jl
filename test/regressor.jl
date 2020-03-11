# Multivariate NN Regressor
# in MLJ multivariate inputs are tables:
N = 200
X = MLJBase.table(randn(10N, 5))
y = rand(2000, 1)
# while multivariate targets are vectors of tuples:
ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5)
y = [Tuple(ymatrix[i,:]) for i in 1:size(ymatrix, 1)]

train = 1:7N
test = (7N+1):10N

se(yhat, y) = sum((yhat .- y).^2)
mse(yhat, y) = mean(broadcast(se, yhat, y))

builder = MLJFlux.Short(σ=identity)
model = MLJFlux.MultivariateNeuralNetworkRegressor(loss=mse, builder=builder)

fitresult, cache, report =
        MLJBase.fit(model, 1, MLJBase.selectrows(X,train), y[train])

# Update model without retraining
model.epochs = 15
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

# Update model by complete retraining
model.batch_size = 2
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))


# univariate targets are ordinary vectors:
y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5

uni_model = MLJFlux.NeuralNetworkRegressor(loss=mse, builder=builder)

fitresult, cache, report =
    MLJBase.fit(uni_model, 1, MLJBase.selectrows(X,train), y[train])

# Update without complete retraining
uni_model.epochs = 15
fitresult, cache, report =
    MLJBase.update(uni_model, 1, fitresult, cache,
                MLJBase.selectrows(X,train), y[train])

# Update model by complete retraining
uni_model.batch_size = 2
fitresult, cache, report =
    MLJBase.update(uni_model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(uni_model, fitresult, MLJBase.selectrows(X, test))
