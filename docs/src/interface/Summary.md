## Models

MLJFlux provides the model types below, for use with input features `X` and targets `y` of
the [scientific type](https://juliaai.github.io/ScientificTypes.jl/dev/)
indicated in the table below. The parameters `n_in`, `n_out` and `n_channels` refer to
information passed to the builder, as described under [Defining Custom Builders](@ref).

| Model Type                                  | Prediction type | `scitype(X) <: _`                                                       | `scitype(y) <: _`                               |
|---------------------------------------------|-----------------|-------------------------------------------------------------------------|-------------------------------------------------|
| [`NeuralNetworkRegressor`](@ref)            | `Deterministic` | `AbstractMatrix{Continuous}` or `Table(Continuous)` with `n_in` columns | `AbstractVector{<:Continuous)` (`n_out = 1`)    |
| [`MultitargetNeuralNetworkRegressor`](@ref) | `Deterministic` | `AbstractMatrix{Continuous}` or `Table(Continuous)` with `n_in` columns | `<: Table(Continuous)` with `n_out` columns     |
| [`NeuralNetworkClassifier`](@ref)           | `Probabilistic` | `AbstractMatrix{Continuous}` or `Table(Continuous)` with `n_in` columns | `AbstractVector{<:Finite}` with `n_out` classes |
| [`NeuralNetworkBinaryClassifier`](@ref)     | `Probabilistic` | `AbstractMatrix{Continuous}` or `Table(Continuous)` with `n_in` columns | `AbstractVector{<:Finite{2}}` (but `n_out = 1`) |
| [`ImageClassifier`](@ref)                   | `Probabilistic` | `AbstractVector(<:Image{W,H})` with `n_in = (W, H)`                     | `AbstractVector{<:Finite}` with `n_out` classes |


```@raw html
<details><summary><b>What exactly is a "model"?</b></summary>
```
In MLJ a *model* is a mutable struct storing hyper-parameters for some
learning algorithm indicated by the model name, and that's all. In
particular, an MLJ model does not store learned parameters.

!!! warning "Difference in Definition"
	In Flux the term "model" has another meaning. However, as all
	Flux "models" used in MLJFLux are `Flux.Chain` objects, we call them
	*chains*, and restrict use of "model" to models in the MLJ sense.

```@raw html
</details>
```

```@raw html
<details open><summary><b>Are oberservations rows or columns?</b></summary>
```

In MLJ the convention for two-dimensional data (tables and matrices) is
**rows=obervations**. For matrices Flux has the opposite convention. If your data is a
matrix with whose column index the observation index, then your optimal solution is to
present the `adjoint` or `transpose` of your matrix to MLJFlux models. Otherwise, you can
use the matrix as is, or transform one time with `permutedims`, and again present the
`adjoint` or `transpose` as the optimal solution for MLJFlux training.

```@raw html
</details>
```

Instructions for coercing common image formats into some `AbstractVector{<:Image}` are
[here](https://juliaai.github.io/ScientificTypes.jl/dev/#Type-coercion-for-image-data).


```@raw html
<details closed><summary><b>Fitting and warm restarts</b></summary>
```
MLJ machines cache state enabling the "warm restart" of model
training, as demonstrated in the incremental training example. In the case of MLJFlux
models, `fit!(mach)` will use a warm restart if:

- only `model.epochs` has changed since the last call; or

- only `model.epochs` or `model.optimiser` have changed since the last
  call and `model.optimiser_changes_trigger_retraining == false` (the
  default) (the "state" part of the optimiser is ignored in this
  comparison). This allows one to dynamically modify learning rates,
  for example.

Here `model=mach.model` is the associated MLJ model.

The warm restart feature makes it possible to externally control iteration. See, for
example, [Early Stopping with MLJFlux](@ref) and [Using MLJ to classifiy the MNIST image
dataset](@ref).

```@raw html
</details>
```



## Model Hyperparameters.

All models share the following hyper-parameters. See individual model docstrings for a full list.

| Hyper-parameter                        | Description                                                                                                                                                                                                                           | Default                                                                                                        |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `builder`                              | Default builder for models.                                                                                                                                                                                                           | `MLJFlux.Linear(σ=Flux.relu)` (regressors) or `MLJFlux.Short(n_hidden=0, dropout=0.5, σ=Flux.σ)` (classifiers) |
| `optimiser`                            | The optimiser to use for training.                                                                                                                                                                                                    | `Optimiser.Adam()`                                                                                                  |
| `loss`                                 | The loss function used for training.                                                                                                                                                                                                  | `Flux.mse` (regressors) and `Flux.crossentropy` (classifiers)                                                  |
| `n_epochs`                             | Number of epochs to train for.                                                                                                                                                                                                        | `10`                                                                                                           |
| `batch_size`                           | The batch size for the data.                                                                                                                                                                                                          | `1`                                                                                                            |
| `lambda`                               | The regularization strength. Range = [0, ∞).                                                                                                                                                                                          | `0`                                                                                                            |
| `alpha`                                | The L2/L1 mix of regularization. Range = [0, 1].                                                                                                                                                                                      | `0`                                                                                                            |
| `rng`                                  | The random number generator (RNG) passed to builders, for weight initialization, for example. Can be any `AbstractRNG` or the seed (integer) for a `Xoshirio` that is reset on every cold restart of model (machine) training. | `GLOBAL_RNG`                                                                                                   |
| `acceleration`                         | Use `CUDALibs()` for training on GPU; default is `CPU1()`.                                                                                                                                                                            | `CPU1()`                                                                                                       |
| `optimiser_changes_trigger_retraining` | True if fitting an associated machine should trigger retraining from scratch whenever the optimiser changes.                                                                                                                          | `false`                                                                                                        |


The classifiers have an additional hyperparameter `finaliser` (default is `Flux.softmax`,
or `Flux.σ` in the binary case) which is the operation applied to the unnormalized output
of the final layer to obtain probabilities (outputs summing to one). It should return a
vector of the same length as its input.

!!! note "Loss Functions"

	Currently, the loss function specified by `loss=...` is applied
	internally by Flux and needs to conform to the Flux API. You cannot,
	for example, supply one of MLJ's probabilistic loss functions, such as
	`MLJ.cross_entropy` to one of the classifier constructors.

That said, you can only use MLJ loss functions or metrics in evaluation meta-algorithms
(such as cross validation) and they will work even if the underlying model comes from
`MLJFlux`.

```@raw html
<details closed><summary><b>More on accelerated training with GPUs</b></summary>
```
As in the table, when instantiating a model for training on a GPU, specify
`acceleration=CUDALibs()`, as in

```julia
using MLJ
ImageClassifier = @load ImageClassifier
model = ImageClassifier(epochs=10, acceleration=CUDALibs())
mach = machine(model, X, y) |> fit!
```

In this example, the data `X, y` is copied onto the GPU under the hood
on the call to `fit!` and cached for use in any warm restart (see
above). The Flux chain used in training is always copied back to the
CPU at then conclusion of `fit!`, and made available as
`fitted_params(mach)`.
```@raw html
</details>
```


## Builders

| Builder                                                       | Description                                                              |
|:--------------------------------------------------------------|:-------------------------------------------------------------------------|
| [`MLJFlux.MLP`](@ref)`(hidden=(10,))`                         | General multi-layer perceptron                                           |
| [`MLJFlux.Short`](@ref)`(n_hidden=0, dropout=0.5, σ=sigmoid)` | Fully connected network with one hidden layer and dropout                |
| [`MLJFlux.Linear`](@ref)`(σ=relu)`                            | Vanilla linear network with no hidden layers and activation function `σ` |
| [`MLJFlux.@builder`](@ref)                                            | Macro for customized builders                                            |
|                                                               |                                                                          |
