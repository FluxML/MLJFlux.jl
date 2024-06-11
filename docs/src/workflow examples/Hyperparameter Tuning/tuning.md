```@meta
EditURL = "tuning.jl"
```

# Hyperparameter Tuning with MLJFlux

In this workflow example we learn how to tune different hyperparameters of MLJFlux models with emphasis on training hyperparameters.

**Julia version** is assumed to be 1.10.*

### Basic Imports

````@example Tuning
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import RDatasets        # Dataset source
using Plots             # To plot tuning results
````

### Loading and Splitting the Data

````@example Tuning
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
X = Float32.(X);      # To be compatible with type of network network parameters
nothing #hide
````

### Instantiating the model
Now let's construct our model. This follows a similar setup the one followed in the [Quick Start](../../index.md#Quick-Start).

````@example Tuning
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Flux.ADAM(0.01),
    batch_size=8,
    epochs=10,
    rng=42
    )
````

### Hyperparameter Tuning Example
Let's tune the batch size and the learning rate. We will use grid search and 5-fold cross-validation.

We start by defining the hyperparameter ranges

````@example Tuning
r1 = range(clf, :batch_size, lower=1, upper=64)
r2 = range(clf, :(optimiser.eta), lower=10^-4, upper=10^0, scale=:log10)
````

Then passing the ranges along with the model and other arguments to the `TunedModel` constructor.

````@example Tuning
tuned_model = TunedModel(
    model=clf,
    tuning=Grid(goal=25),
    resampling=CV(nfolds=5, rng=42),
    range=[r1, r2],
    measure=cross_entropy,
);
nothing #hide
````

Then wrapping our tuned model in a machine and fitting it.

````@example Tuning
mach = machine(tuned_model, X, y);
fit!(mach, verbosity=0);
nothing #hide
````

Let's check out the best performing model:

````@example Tuning
fitted_params(mach).best_model
````

We can visualize the hyperparameter search results as follows

````@example Tuning
plot(mach)
````

### Learning Curves
With learning curves, it's possible to center our focus on the effects of a single hyperparameter of the model

First define the range and wrap it in a learning curve

````@example Tuning
r = range(clf, :epochs, lower=1, upper=200, scale=:log10)
curve = learning_curve(clf, X, y,
                       range=r,
                       resampling=CV(nfolds=4, rng=42),
                       measure=cross_entropy)
````

Then plot the curve

````@example Tuning
plot(curve.parameter_values,
       curve.measurements,
       xlab=curve.parameter_name,
       xscale=curve.parameter_scale,
       ylab = "Cross Entropy")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

