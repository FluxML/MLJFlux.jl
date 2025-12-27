```@meta
EditURL = "notebook.jl"
```

# Early Stopping with MLJ

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/early_stopping).

In this workflow example, we learn how MLJFlux enables us to easily use early stopping
when training MLJFlux models.

**This script tested using Julia 1.10**

### Basic Imports

````@example early_stopping
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
using Plots             # To visualize training
import Optimisers       # native Flux.jl optimisers no longer supported
using StableRNGs        # for reproducibility across Julia versions

stable_rng() = StableRNGs.StableRNG(123)
````

### Loading and Splitting the Data

````@example early_stopping
iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
X = fmap(column-> Float32.(column), X) # Flux prefers Float32 data
````

### Instantiating the model Now let's construct our model. This follows a similar setup
to the one followed in the [Quick Start](../../index.md#Quick-Start).

````@example early_stopping
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=50,
    rng=stable_rng(),
)
````

### Wrapping it in an IteratedModel

Let's start by defining the condition that can cause the model to early stop.

````@example early_stopping
stop_conditions = [
    Step(1),            # Repeatedly train for one iteration
    NumberLimit(100),   # Don't train for more than 100 iterations
    Patience(5),        # Stop after 5 iterations of disimprovement in validation loss
    NumberSinceBest(9), # Or if the best loss occurred 9 iterations ago
    TimeLimit(30/60),   # Or if 30 minutes passed
]
````

We can also define callbacks. Here we want to store the validation loss for each iteration

````@example early_stopping
validation_losses = []
callbacks = [
    WithLossDo(loss->push!(validation_losses, loss)),
]
````

Construct the iterated model and pass to it the stop_conditions and the callbacks:

````@example early_stopping
iterated_model = IteratedModel(
    model=clf,
    resampling=Holdout(fraction_train=0.7); # loss and stopping are based on out-of-sample
    measures=log_loss,
    iteration_parameter=:(epochs),
    controls=vcat(stop_conditions, callbacks),
    retrain=false            # no need to retrain on all data at the end
);
nothing #hide
````

You can see more advanced stopping conditions as well as how to involve callbacks in the
[documentation](https://juliaai.github.io/MLJ.jl/stable/controlling_iterative_models/#Controlling-Iterative-Models)

### Training with Early Stopping

At this point, all we need is to fit the model and iteration controls will be
automatically handled

````@example early_stopping
mach = machine(iterated_model, X, y)
fit!(mach)
# We can get the training losses like so
training_losses = report(mach).model_report.training_losses;
nothing #hide
````

### Results

We can see that the model converged after 100 iterations.

````@example early_stopping
plot(training_losses, label="Training Loss", linewidth=2)
plot!(validation_losses, label="Validation Loss", linewidth=2, size=(800,400))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

