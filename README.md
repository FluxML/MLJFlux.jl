# MLJFlux 

An interface to Flux deep learning models for the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine learning framework

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJFlux.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJFlux.jl) [![Coverage Status](https://coveralls.io/repos/github/alan-turing-institute/MLJFlux.jl/badge.svg?branch=master)](https://coveralls.io/github/alan-turing-institute/MLJFlux.jl?branch=master)

MLJFlux.jl makes a variety of deep learning models available to users
of the MLJ machine learning toolbox by providing an interface to
[Flux](https://github.com/FluxML/Flux.jl) framework.

This package is a work-in-progess and does not have a stable
API. Presently, the user should be familiar with building a Flux
chain.

### Models

In MLJ a *model* is a mutable struct storing hyperparameters for some learning algorithm indicated by the model name. MLJFlux provides three such models:

- `NeuralNetworkRegressor`
- `MultivariateNeuralNetworkRegressor`
- `NeuralNetworkClassifier`
- `ImageClassifier`

*Warning:* In Flux the term "model" has another meaning. However, as all
Flux "models" used in MLJFLux are `Flux.Chain` objects, we call them
*chains*, and restrict use of "model" to models in the MLJ sense.


### Constructing a model

Construction begins by defining an auxiliary struct called a
*builder*, and an associated `fit` method, for generating a
`Flux.Chain` object compatible with the data (bound later to the MLJ
model). The struct must be derived from MLJFlux.Builder, as in this
example:

```julia
mutable struct MyNetwork <: MLJFlux.Builder
    n1 :: Int
    n2 :: Int
end

function MLJFlux.fit(nn::MyNetwork, a, b)
    return Chain(Dense(a, nn.n1), Dense(nn.n1, nn.n2), Dense(nn.n2, b))
end
```

*Notes:*

- The attributes of the MyNetwork struct `n1`, `n2` can be anything. What matters is the result of the `fit` function.
- Here `a` is the the number of input features, inferred from
  the data by MLJ when the model is trained. (It may be this argument is ignored, as in an
  initial convolution layer for image classification).
- Here `b` is the dimension of the target variable
  (`NeuralNetworkRegressor`) or the number of (univariate) target
   levels (`NeuralNetworkClassifier` or `ImageClassifier`) - again inferred from the data. 

Now that we have a builder, we can instantiate an MLJ model. For example:

```julia
nn_regressor = NeuralNetworkRegressor(builder=MyNetwork(32, 16), 
loss=Flux.mse, epochs=5)
```

The object `nn_regressor` behaves like any other MLJ model. It can be wrapped inside an MLJ `machine`, and you can do anything you'd do with
an MLJ machine.

```julia
mach = machine(nn_regressor, X, y)
fit!(mach, verbosity=2)
yhat = predict(mach, rows = train)
```
and so on.


### Loss functions.

The loss function specified by `loss=...` is applied internally by
Flux and needs to conform to the Flux API. You cannot, for example,
supply one of MLJ's probablistic loss functions, such as
`MLJ.cross_entropy` to one of the classifiers.  Unless, you are
familiar with this API, it is recommended you use one of the [loss
functions provided by
Flux](https://github.com/FluxML/Flux.jl/blob/v0.8.3/src/layers/stateless.jl)
	or leave `loss` unspecified to invoke the default. For a binary classification problem you might also consider `Flux.binarycrossentropy`, while for a classification problem with more than two classes (most image problems) consider `Flux.logitbinarycrossentropy`, as these have better numerical stability than vanilla `Flux.crossentropy`. 


### Hyperparameters.

`NeuralNetworkRegressor` and `NeuralNetworkClassifier` have the following hyperparameters:

1. `builder`: An instance of some concrete subtype of
   `MLJFlux.Builder`, as in the above example
    
2. `optimiser`: The optimiser to use for training. Default =
   `Flux.ADAM()`

3. `loss`: The loss function used for training. Default = `Flux.mse` (regressors) and `Flux.crossentropy` (classifiers)

4. `n_epochs`: Number of epochs to train for. Default = `10`

5. `batch_size`: The batch_size for the data. Default = 1

6. `lambda`: The regularization strength. Default = 0. Range = [0, ∞)

7. `alpha`: The L2/L1 mix of regularization. Default = 0. Range = [0, 1] 

8. `optimiser_changes_trigger_retraining`: True if fitting an
   associated machine should trigger retraining from scratch whenever
   the optimiser changes. Default = false

<!-- 9. `embedding_choice`: The embedding to use for handling categorical features. Options = :onehot, :entity_embedding. Default = :onehot. -->

<!-- 10. `embedding_dimension`: Valid only when -->
<!--     `embedding_choice=:entity_embedding`. The dimension follows the -->
<!--     formula `min(embedding_dimension, levels)`, where levels is the -->
<!--     number of levels in the pool of the categorical feature. If the -->
<!--     value is <= 0, this means that the dimension will be equal to (the -->
<!--     number of unique values of the feature) / 2. Default = -1 -->
