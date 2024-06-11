# # Incremental Training with MLJFlux

# In this workflow example we explore how to incrementally train MLJFlux models.

using Pkg     #!md
Pkg.activate(@__DIR__);     #!md
Pkg.instantiate();     #!md

# **Julia version** is assumed to be 1.10.* This tutorial is available as a Jupyter
# notebook or julia script
# [here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/workflow_examples/incremental_training).

# ### Basic Imports

using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import RDatasets        # Dataset source
import Optimisers       # native Flux.jl optimisers no longer supported


# ### Loading and Splitting the Data

iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
X = Float32.(X)      # To be compatible with type of network network parameters
(X_train, X_test), (y_train, y_test) = partition(
    (X, y), 0.8,
    multi = true,
    shuffle = true,
    rng=42,
);


# ### Instantiating the model

# Now let's construct our model. This follows a similar setup to the one followed in the
# [Quick Start](../../index.md#Quick-Start).

NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=10,
    rng=42,
)

# ### Initial round of training

# Now let's train the model. Calling fit! will automatically train it for 100 epochs as
# specified above.

mach = machine(clf, X_train, y_train)
fit!(mach)

# Let's evaluate the training loss and validation accuracy
training_loss = cross_entropy(predict(mach, X_train), y_train)

#-

val_acc = accuracy(predict_mode(mach, X_test), y_test)

# Poor performance it seems.

# ### Incremental Training

# Now let's train it for another 30 epochs at half the original learning rate. All we need
# to do is changes these hyperparameters and call fit again. It won't reset the model
# parameters before training.

clf.optimiser = Optimisers.Adam(clf.optimiser.eta/2)
clf.epochs = clf.epochs + 30
fit!(mach, verbosity=2);

# Let's evaluate the training loss and validation accuracy

training_loss = cross_entropy(predict(mach, X_train), y_train)

#-

training_acc = accuracy(predict_mode(mach, X_test), y_test)

#-

# That's much better. If we are rather interested in resetting the model parameters before
# fitting, we can do `fit(mach, force=true)`.
