# Multitarget NN Regressor

@testset "collate" begin

    # NeuralNetworRegressor:
    Xmatrix = rand(10, 3)
    X = MLJBase.table(Xmatrix)
    y = rand(10)
    model = MLJFlux.NeuralNetworkRegressor()
    batch_size= 3
    @test MLJFlux.collate(model, X, y, batch_size) ==
        [(Xmatrix'[:,1:3], y[1:3]),
         (Xmatrix'[:,4:6], y[4:6]),
         (Xmatrix'[:,7:9], y[7:9]),
         (Xmatrix'[:,10:10], y[10:10])]

    # MultitargetNeuralNetworRegressor:
    ymatrix = rand(10, 2)
    y = MLJBase.table(ymatrix)
    model = MLJFlux.NeuralNetworkRegressor()
    batch_size= 3
    @test MLJFlux.collate(model, X, y, batch_size) ==
        [(Xmatrix'[:,1:3], ymatrix'[:,1:3]),
         (Xmatrix'[:,4:6], ymatrix'[:,4:6]),
         (Xmatrix'[:,7:9], ymatrix'[:,7:9]),
         (Xmatrix'[:,10:10], ymatrix'[:,10:10])]
end

N = 200
X = MLJBase.table(randn(10N, 5))

# multitargets are tables:
ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5)
y = Tables.table(ymatrix)

train = 1:7N
test = (7N+1):10N

se(yhat, y) = sum((yhat .- y).^2)
mse(yhat, y) = mean(broadcast(se, yhat, y))

builder = MLJFlux.Short(σ=identity)
model = MLJFlux.MultitargetNeuralNetworkRegressor(loss=mse, builder=builder)

fitresult, cache, report =
    MLJBase.fit(model, 1, MLJBase.selectrows(X,train), MLJBase.selectrows(y, train))

# Update model without retraining
model.epochs = 15
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), MLJBase.selectrows(y, train))

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

# Update model by complete retraining
model.batch_size = 2
fitresult, cache, report =
    MLJBase.update(model, 1, fitresult, cache,
                   MLJBase.selectrows(X,train), MLJBase.selectrows(y, train))

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

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
