# # Incremental Training with MLJFlux

# This demonstration is available as a Jupyter notebook or julia script
# [here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/incremental_training).

# In this workflow example we explore how to incrementally train MLJFlux models.

using Pkg     #!md
PKG_ENV = joinpath(@__DIR__, "..", "..", "..")
Pkg.activate(PKG_ENV);     #!md
Pkg.instantiate();     #!md

# **Julia version** is assumed to be 1.10.*

# ### Basic Imports

using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import Optimisers       # native Flux.jl optimisers no longer supported
using StableRNGs        # for reproducibility across Julia versions

stable_rng() = StableRNGs.StableRNG(123)


# ### Loading and Splitting the Data

iris = load_iris() # a named-tuple of vectors
y, X = unpack(iris, ==(:target), rng=stable_rng())
X = fmap(column-> Float32.(column), X) # Flux prefers Float32 data
(X_train, X_test), (y_train, y_test) = partition(
    (X, y), 0.8,
    multi = true,
    shuffle = true,
    rng=stable_rng(),
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
    rng=stable_rng(),
)

# ### Initial round of training

# Now let's train the model. Calling fit! will automatically train it for 100 epochs as
# specified above.

mach = machine(clf, X_train, y_train)
fit!(mach, verbosity=0)

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
