```@meta
EditURL = "notebook.jl"
```

# Hyperparameter Tuning with MLJFlux

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/hyperparameter_tuning).

In this workflow example we learn how to tune different hyperparameters of MLJFlux
models with emphasis on training hyperparameters.

````@example hyperparameter_tuning
PKG_ENV = joinpath(@__DIR__, "..", "..", "..")
````

**This script tested using Julia 1.10**

### Basic Imports

````@example hyperparameter_tuning
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
using Plots             # To plot tuning results
import Optimisers       # native Flux.jl optimisers no longer supported
using StableRNGs        # for reproducibility across Julia versions

stable_rng() = StableRNGs.StableRNG(123)
````

### Loading and Splitting the Data

````@example hyperparameter_tuning
iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
X = fmap(column-> Float32.(column), X) # Flux prefers Float32 data
````

### Instantiating the model

Now let's construct our model. This follows a similar setup the one followed in the
[Quick Start](../../index.md#Quick-Start).

````@example hyperparameter_tuning
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=10,
    rng=stable_rng(),
)
````

### Hyperparameter Tuning Example

Let's tune the batch size and the learning rate. We will use grid search and 5-fold
cross-validation.

We start by defining the hyperparameter ranges

````@example hyperparameter_tuning
r1 = range(clf, :batch_size, lower=1, upper=64)
etas = [10^x for x in range(-4, stop=0, length=4)]
optimisers = [Optimisers.Adam(eta) for eta in etas]
r2 = range(clf, :optimiser, values=optimisers)
````

Then passing the ranges along with the model and other arguments to the `TunedModel`
constructor.

````@example hyperparameter_tuning
tuned_model = TunedModel(
    model=clf,
    tuning=Grid(goal=25),
    resampling=CV(nfolds=5, rng=stable_rng()),
    range=[r1, r2],
    measure=cross_entropy,
);
nothing #hide
````

Then wrapping our tuned model in a machine and fitting it.

````@example hyperparameter_tuning
mach = machine(tuned_model, X, y);
fit!(mach, verbosity=0);
nothing #hide
````

Let's check out the best performing model:

````@example hyperparameter_tuning
fitted_params(mach).best_model
````

### Learning Curves

With learning curves, it's possible to center our focus on the effects of a single
hyperparameter of the model

First define the range and wrap it in a learning curve

````@example hyperparameter_tuning
r = range(clf, :epochs, lower=1, upper=200, scale=:log10)
curve = learning_curve(
    clf,
    X,
    y,
    range=r,
    resampling=CV(nfolds=4, rng=stable_rng()),
    measure=cross_entropy,
)
````

Then plot the curve

````@example hyperparameter_tuning
plot(
    curve.parameter_values,
    curve.measurements,
    xlab=curve.parameter_name,
    xscale=curve.parameter_scale,
    ylab = "Cross Entropy",
)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

