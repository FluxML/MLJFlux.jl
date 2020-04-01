## NEURAL NETWORK CLASSIFIER

N = 100
X = MLJBase.table(randn(10N, 5))

train = 1:7N
test = (7N+1):10N

y = CategoricalArray(rand("abcd", 1000));

builder = MLJFlux.Linear(σ=Flux.sigmoid)
model = MLJFlux.NeuralNetworkClassifier(loss=Flux.crossentropy, builder=builder)

fitresult, cache, report =
    MLJBase.fit(model, 2, MLJBase.selectrows(X,train), y[train])

yhat = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))

# Update without completely retraining
model.epochs = 15
fitresult, cache, report =
    MLJBase.update(model, 2, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

# Update by completely retraining
model.batch_size = 5
fitresult, cache, report =
    MLJBase.update(model, 3, fitresult, cache,
                   MLJBase.selectrows(X,train), y[train])

true
