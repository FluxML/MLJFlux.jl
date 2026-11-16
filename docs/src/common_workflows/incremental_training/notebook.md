```@meta
EditURL = "notebook.jl"
```

# Incremental Training with MLJFlux

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/incremental_training).

In this workflow example we explore how to incrementally train MLJFlux models.

**Julia version** is assumed to be at least 1.12.

### Basic Imports

````@example incremental_training
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import Optimisers       # native Flux.jl optimisers no longer supported
using StableRNGs        # for reproducibility across Julia versions
stable_rng() = StableRNGs.StableRNG(123);
nothing #hide
````

### Loading and Splitting the Data

````@example incremental_training
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

Now let's construct our model. This follows a similar setup to the one followed in the
[Quick Start](@ref).

````@example incremental_training
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    eta = 0.01,
    batch_size=8,
    epochs=10,
    rng=stable_rng(),
)
````

Instead of explicitly specifying an optimiser, we are going with the default (which is
`Adam(eta=0.001, beta=(0.9, 0.999), epsilon=1.0e-8)`) and specifying a learning rate
override of `eta=0.01` (equivalent to specifying `optImiser=Adam(0.01)`).

### Initial round of training

Now let's train the model. Calling fit! will automatically train it for 100 epochs as
specified above.

````@example incremental_training
mach = machine(clf, X_train, y_train)
fit!(mach, verbosity=0)
````

Let's evaluate the training loss and validation accuracy

````@example incremental_training
training_loss = cross_entropy(predict(mach, X_train), y_train)
````

````@example incremental_training
val_acc = accuracy(predict_mode(mach, X_test), y_test)
````

Poor performance it seems.

### Incremental Training

Now let's train it for another 30 epochs at half the original learning rate. All we need
to do is change these hyperparameters and call `fit!` again. The learned
model parameters (weights and biases) are not re-initialized.

````@example incremental_training
clf.eta = clf.eta*0.5
clf.epochs = clf.epochs + 30
fit!(mach, verbosity=2);
nothing #hide
````

If you'd rather changes to `eta` or `optimiser` force a cold restart, then set the
`optimiser_changes_trigger_retraining` hyperparameter to `true`. You might want do this
when comparing the effect of different learning rates on the final performance, using
MLJ's `TunedModel` wrapper, for example.

Let's evaluate the training loss and
validation accuracy

````@example incremental_training
training_loss = cross_entropy(predict(mach, X_train), y_train)
````

````@example incremental_training
training_accuracy = accuracy(predict_mode(mach, X_test), y_test)
````

That's much better. If we are rather interested in resetting the model parameters before
fitting, we can do `fit(mach, force=true)`.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

