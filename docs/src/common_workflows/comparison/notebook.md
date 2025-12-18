```@meta
EditURL = "notebook.jl"
```

# Model Comparison with MLJFlux

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/comparison).

In this workflow example, we see how we can compare different machine learning models
with a neural network from MLJFlux.

````@example comparison
PKG_ENV = joinpath(@__DIR__, "..", "..", "..")
````

**This script tested using Julia 1.10**

### Basic Imports

````@example comparison
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
using DataFrames        # To visualize hyperparameter search results
import Optimisers       # native Flux.jl optimisers no longer supported
using Measurements       # to get ± functionality
import CategoricalArrays.unwrap
using StableRNGs        # for reproducibility across Julia versions

stable_rng() = StableRNG(123)
````

### Loading and Splitting the Data

````@example comparison
iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
````

### Instantiating the models Now let's construct our model. This follows a similar setup
to the one followed in the [Quick Start](../../index.md#Quick-Start).

````@example comparison
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

clf1 = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=50,
    rng=stable_rng(),
    )
````

Let's as well load and construct three other classical machine learning models:

````@example comparison
BayesianLDA = @load BayesianLDA pkg=MultivariateStats
clf2 = BayesianLDA()
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
clf3 = RandomForestClassifier()
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost
clf4 = XGBoostClassifier();
nothing #hide
````

### Wrapping One of the Models in a TunedModel

Instead of just comparing with four models with the default/given hyperparameters, we
will give `XGBoostClassifier` an unfair advantage By wrapping it in a `TunedModel` that
considers the best learning rate η for the model.

````@example comparison
r1 = range(clf4, :eta, lower=0.01, upper=0.5, scale=:log10)
tuned_model_xg = TunedModel(
    model=clf4,
    ranges=[r1],
    tuning=Grid(resolution=10),
    resampling=CV(nfolds=5, rng=stable_rng()),
    measure=cross_entropy,
);
nothing #hide
````

Of course, one can wrap each of the four in a TunedModel if they are interested in
comparing the models over a large set of their hyperparameters.

### Comparing the models

We simply pass the four models to the `models` argument of the `TunedModel` construct

````@example comparison
tuned_model = TunedModel(
    models=[clf1, clf2, clf3, tuned_model_xg],
    tuning=Explicit(),
    resampling=CV(nfolds=2, rng=stable_rng()),
    repeats=5,
    measure=cross_entropy,
);
nothing #hide
````

Notice here we are using 5 x 2 Monte Carlo cross-validation.

Then wrapping our tuned model in a machine and fitting it.

````@example comparison
mach = machine(tuned_model, X, y);
fit!(mach, verbosity=0);
nothing #hide
````

Now let's see the history for more details on the performance for each of the models

````@example comparison
history = report(mach).history
history_df = DataFrame(
    mlp = [x.model for x in history],
    measurement = [
        x.evaluation.measurement[1] ±
            x.evaluation.uncertainty_radius_95[1] for x in history
                ],
)
sort!(history_df, [order(:measurement)])
````

This is Occam's razor in practice.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

