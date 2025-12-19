```@meta
EditURL = "notebook.jl"
```

# Live Training with MLJFlux

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/live_training).

````@example live_training
PKG_ENV = joinpath(@__DIR__, "..", "..", "..")
````

**This script tested using Julia 1.10**

### Basic Imports

````@example live_training
using MLJ
using Flux
import Optimisers
using StableRNGs        # for reproducibility across Julia versions

stable_rng() = StableRNGs.StableRNG(123)
````

````@example live_training
using Plots
````

### Loading and Splitting the Data

````@example live_training
iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
X = fmap(column-> Float32.(column), X) # Flux prefers Float32 data
````

### Instantiating the model

Now let's construct our model. This follows a similar setup to the one followed in the
[Quick Start](../../index.md#Quick-Start).

````@example live_training
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=50,
    rng=stable_rng(),
)
````

Now let's wrap this in an iterated model. We will use a callback that makes a plot for
validation losses each iteration.

````@example live_training
stop_conditions = [
    Step(1),            # Repeatedly train for one iteration
    NumberLimit(100),   # Don't train for more than 100 iterations
]

validation_losses =  []
gr(reuse=true)                  # use the same window for plots
function plot_loss(loss)
    push!(validation_losses, loss)
    display(plot(validation_losses, label="validation loss", xlim=(1, 100)))
    sleep(.01)  # to catch up with the plots while they are being generated
end

callbacks = [ WithLossDo(plot_loss),]

iterated_model = IteratedModel(
    model=clf,
    resampling=Holdout(),
    measures=log_loss,
    iteration_parameter=:(epochs),
    controls=vcat(stop_conditions, callbacks),
    retrain=true,
)
````

### Live Training
Simply fitting the model is all we need

````@example live_training
mach = machine(iterated_model, X, y)
fit!(mach)
validation_losses
````

Note that the wrapped model sets aside some data on which to make out-of-sample
estimates of the loss, which is how `validation_losses` are calculated. But if we use
`mach` to make predictions on new input features, these are based on retraing the model
on *all* provided data.

````@example live_training
Xnew = (
    sepal_length = Float32[5.8, 5.8, 5.8],
    sepal_width = Float32[4.0, 2.6, 2.7],
    petal_length = Float32[1.2, 4.0, 4.1],
    petal_width = Float32[0.2, 1.2, 1.0],
)

predict_mode(mach, Xnew)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

