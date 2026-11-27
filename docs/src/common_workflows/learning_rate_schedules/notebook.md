```@meta
EditURL = "notebook.jl"
```

# Implementing a Learning Rate Schedule

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/learning_rate_schedules).

A **learning rate schedule** describes dynamic changes to the learning rate as training
proceeds.  In MLJ such schedules are realized using MLJ's `IteratedModel` wrapper. Here
we demonstrate exponential decay, step decay, and cyclic learning rates.

In this illustration, no attempt has been made to optimize the schedule parameters.

**Julia version** is assumed to be at least 1.12.

### Basic Imports

````@example learning_rate_schedules
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import Optimisers       # native Flux.jl optimisers no longer supported
using StableRNGs        # for reproducibility across Julia versions

stable_rng() = StableRNGs.StableRNG(123);
nothing #hide
````

### Loading and Splitting the Data

````@example learning_rate_schedules
iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
X = fmap(column-> Float32.(column), X) # Flux prefers Float32 data
(X_train, X_test), (y_train, y_test) = partition(
    (X, y), 0.8,
    multi = true,
    shuffle = true,
    rng=stable_rng(),
);
nothing #hide
````

### Instantiating the model

Now let's construct our model, as we did for [Incremental Training with
MLJFlux](@ref):

````@example learning_rate_schedules
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    eta = 0.05,
    batch_size=8,
    epochs=10,
    rng=stable_rng(),
)
````

To implement the learning rate schedule, we wrap our model using MLJ's
[`IteratedModel`](https://juliaai.github.io/MLJ.jl/dev/controlling_iterative_models/#Controlling-Iterative-Models)
wrapper, including an appropriate `WithModelDo` callback control. Since we are using the
wrapper, we might as well add early stopping criteria.

````@example learning_rate_schedules
decay(factor) = WithModelDo(model -> (model.eta *= factor))
controls = [
    Step(1),                        # train one epoch for each control cycle
    decay(0.95),                    # reduce learning rate by scaling by 0.95 each cycle
    Patience(3),                    # Stop when the loss drops below 0.001
    WithLossDo(loss -> @info "loss: $loss"),       # print the loss
    InvalidValue(),                 # Stop if NaN or Inf encountered
    NumberLimit(100),               # Stop after 100 control cycles
]
iterated_clf = IteratedModel(clf; controls, resampling=nothing);
nothing #hide
````

We use `resampling=nothing` here to signal that all supplied data should be used for
training and that the training loss (as reported by the model) should be the basis for
the stopping criteria. To use an out-of-sample loss instead, specify
`resampling=Holdout()`.

````@example learning_rate_schedules
mach = machine(iterated_clf, X_train, y_train)
fit!(mach);
nothing #hide
````

We can predict using the wrapped model as we would the original model, as here:

````@example learning_rate_schedules
test_loss = cross_entropy(predict(mach, X_test), y_test)
````

To get step decay, we can use the `skip` control wrapper:

````@example learning_rate_schedules
step_decay(factor, width) = MLJIteration.skip(decay(factor), predicate=width);
nothing #hide
````

For example, `step_decay(0.7, 10)` will multiply the learning rate by 0.7 every 10
epochs:

````@example learning_rate_schedules
controls = [
    Step(1),                        # train one epoch for each control cycle
    step_decay(0.7, 10),
    Patience(3),                    # Stop when the loss drops below 0.001
    WithLossDo(loss -> @info "loss: $loss"),       # print the loss
    InvalidValue(),                 # Stop if NaN or Inf encountered
    NumberLimit(100),               # Stop after 100 control cycles
]
iterated_clf = IteratedModel(clf; controls, resampling=nothing)
mach = machine(iterated_clf, X_train, y_train) |> fit!;
nothing #hide
````

````@example learning_rate_schedules
test_loss = cross_entropy(predict(mach, X_test), y_test)
````

While cyclic learning rates can be implemented using a more complicated variation on the
above pattern, a built-in control is provided for convenience. Consult the
`CycleLearningRate` document string for details and some differences to be found in
other implementations.

````@example learning_rate_schedules
cycle = CycleLearningRate(stepsize=4, lower=0.001, upper=0.05, learning_rate_parameter=:eta)
controls = [
    Step(1),                        # train one epoch for each control cycle
    cycle,
    Patience(3),                    # Stop when the loss drops below 0.001
    WithLossDo(loss -> @info "loss: $loss"),       # print the loss
    InvalidValue(),                 # Stop if NaN or Inf encountered
    NumberLimit(100),               # Stop after 100 control cycles
]
iterated_clf = IteratedModel(clf; controls, resampling=nothing)
mach = machine(iterated_clf, X_train, y_train) |> fit!;
nothing #hide
````

````@example learning_rate_schedules
test_loss = cross_entropy(predict(mach, X_test), y_test)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

