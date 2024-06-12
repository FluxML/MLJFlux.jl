```@meta
EditURL = "notebook.jl"
```

# Live Training with MLJFlux

This tutorial is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/live_training).

**Julia version** is assumed to be 1.10.*

### Basic Imports

````@example live_training
using MLJ
using Flux
import RDatasets
import Optimisers
````

````@example live_training
using Plots
````

### Loading and Splitting the Data

````@example live_training
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
X = Float32.(X);      # To be compatible with type of network network parameters
nothing #hide
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
    rng=42,
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
fit!(mach, force=true)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

