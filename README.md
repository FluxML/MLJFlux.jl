## MLJFlux : A port to the Flux deep learning framework from MLJ

MLJFlux.jl provides an interface to implement deep learing models with the MLJ model spec. The deep learning part - is handled by the [Flux](https://github.com/FluxML/Flux.jl) deep learning framework.

### Models:

A model in terms of MLJ is basically a container of hyper-parameters. MLJFlux provides 3 such models:

1. NeuralNetworkRegressor
2. NeuralNetworkClassifier
3. ImageClassifier

### Writing your own regressor model:

The first step towards writing your model is defining a struct for the model and the corresponding `fit` function. The `fit` function returns the `Flux.Chain` object.

Note: The struct must be derived from MLJFlux.Builder

```
mutable struct MyNetwork <: MLJFlux.Builder
    n1 :: Int
    n2 :: Int
end

function MLJFlux.fit(nn::MyNetwork, a, b)
    return Chain(Dense(a, nn.n1), Dense(nn.n1, nn.n2), Dense(nn.n2, b))
end
```

Some things to note:
1. While we're doing this for a regressor, the same steps apply for the classifier/image-classifier models too.
2. The attibutes of the MyNetwork struct (n1, n2) can be anything. What matters is the result of the `fit` function.

Now that we have our builder, we can instantiate a regressor object with this as our builder.

```
nn_regressor = NeuralNetworkRegressor(builder=MyNetwork(32, 16), loss=Flux.mse, n=5)
```

(For classification, use `NeuralNetworkClassifier` instead of `NeuralNetworkRegerssor`)

`nn_regressor` is now any other MLJ model. This can be wrapped inside an MLJ `machine`, and you can do anything you'd do with 
an MLJ machine.

```
mach = machine(nn_regressor, X, y)
fit!(mach, verbosity=2)
yhat = predict(mach, rows = train)
```
and so on.

### Hyper-parameters.

NeuralNetworkRegressor / NeuralNetworkClassifier have a few hyperparameters:

1. builder = An instance of the MyNetwork struct from the above example.
2. optimiser = The optimiser to use for training. Default = Flux.ADAM()
3. loss = The loss function used. Default = Flux.mse
4. n = Number of epochs to train for. Default = 10
5. batch_size = The batch_size for the data. Default = 1
6. lambda = Regularization parameter. Default = 0
7. alpha = Regularization parameter. Default = 0
8. optimiser_changes_trigger_retraining = Should changing the optimiser re-train the model. Default = false
9. Embedding choice = The embedding to use for handling categorical features. Options = :onehot, :entity_embedding. Default = :onehot.
10. Embedding dimension = Valid only when `embedding_choice = :entity_embedding`. The dimension is follows the formula `min(embedding_dimension, levels)`, where levels is the number of unique values in the feature. Default = 4.