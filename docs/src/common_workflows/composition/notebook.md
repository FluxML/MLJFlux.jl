```@meta
EditURL = "notebook.jl"
```

# Model Composition with MLJFlux

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/composition).

In this workflow example, we see how MLJFlux enables composing MLJ models with MLJFlux
models. We will assume a class imbalance setting and wrap an oversampler with a deep
learning model from MLJFlux.

**This script tested using Julia 1.10**

### Basic Imports

````@example composition
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import Random           # To create imbalance
import Imbalance        # To solve the imbalance
import Optimisers       # native Flux.jl optimisers no longer supported
using StableRNGs        # for reproducibility across Julia versions
import CategoricalArrays.unwrap

stable_rng() = StableRNGs.StableRNG(123)
````

### Loading and Splitting the Data

````@example composition
iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
X = fmap(column-> Float32.(column), X) # Flux prefers Float32 data
````

The iris dataset has a target with uniformly distributed values, `"versicolor"`,
`"setosa"`, and `"virginica"`. To manufacture an unbalanced dataset, we'll combine the
first two into a single classs, `"colosa"`:

````@example composition
y = coerce(
        map(y) do species
            species == "virginica" ? unwrap(species) : "colosa"
        end,
        Multiclass,
);
Imbalance.checkbalance(y)
````

### Instantiating the model

Let's load `BorderlineSMOTE1` to oversample the data and `Standardizer` to standardize
it.

````@example composition
BorderlineSMOTE1 = @load BorderlineSMOTE1 pkg=Imbalance verbosity=0
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
````

We didn't need to load Standardizer because it is a local model for MLJ (see
`localmodels()`)

````@example composition
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=50,
    rng=stable_rng(),
)
````

First we wrap the oversampler with the neural network via the `BalancedModel`
construct. This comes from `MLJBalancing` And allows combining resampling methods with
MLJ models in a sequential pipeline.

````@example composition
oversampler = BorderlineSMOTE1(k=5, ratios=1.0, rng=stable_rng())
balanced_model = BalancedModel(model=clf, balancer1=oversampler)
standarizer = Standardizer()
````

Now let's compose the balanced model with a standardizer.

````@example composition
pipeline = standarizer |> balanced_model
````

By this, any training data will be standardized then oversampled then passed to the
model. Meanwhile, for inference, the standardizer will automatically use the training
set's mean and std and the oversampler will be play no role.

### Training the Composed Model

The pipeline model can be evaluated like any other model:

````@example composition
mach = machine(pipeline, X, y)
fit!(mach)
cv=CV(nfolds=5)
evaluate!(mach, resampling=cv, measure=accuracy)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

