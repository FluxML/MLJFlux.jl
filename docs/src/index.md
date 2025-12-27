# MLJFlux.jl

A Julia package integrating deep learning Flux models with [MLJ](https://juliaai.github.io/MLJ.jl/dev/).

## Objectives

- Provide a user-friendly and high-level interface to fundamental [Flux](https://fluxml.ai/Flux.jl/stable/) deep learning models while still being extensible by supporting custom models written with Flux

- Make building deep learning models more convenient to users already familiar with the MLJ workflow

- Make it easier to apply machine learning techniques provided by MLJ, including: out-of-sample performance evaluation, hyper-parameter optimization, iteration control, and more, to deep learning models

!!! note "MLJFlux Scope"

    MLJFlux support is focused on fundamental deep learning models for common
    supervised learning tasks. Sophisticated architectures and approaches, such as online
    learning, reinforcement learning, and adversarial networks, are currently outside its
    scope. Also, MLJFlux is limited to tasks where all (batches of) training data
        fits into memory.

## Installation

```julia
import Pkg
Pkg.activate("my_environment", shared=true)
Pkg.add(["MLJ", "MLJFlux", "Optimisers", "Flux"])
```

## Quick Start

```@example
using MLJ, MLJFlux
import Flux

# 1. Load Data
iris = load_iris() # a named-tuple of vectors (but most tables work here)
y, X = unpack(iris, ==(:target), rng=123)
X = Flux.fmap(column-> Float32.(column), X) # Flux prefers Float32 data

# 2. Load and instantiate model
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg="MLJFlux"
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Flux.Adam(0.01),
    batch_size=8,
    epochs=100,
    acceleration=CPU1() # the default; use instead `CUDALibs()` for GPU
    )

# 3. Wrap it in a machine
mach = machine(clf, X, y)

# 4. Evaluate the model
evaluate!(mach, resampling=CV(nfolds=3), repeats=2, measure=[brier_loss, accuracy])

# 5. Fit and predict on new data
fit!(mach)
Xnew = (
    sepal_length = [7.2, 4.4, 5.6],
    sepal_width = [3.0, 2.9, 2.5],
    petal_length = [5.8, 1.4, 3.9],
    petal_width = [1.6, 0.2, 1.1],
)
predict(mach, Xnew)
```

As you can see we are able to use MLJ meta-functionality (in this case Monte Carlo
cross-validation) with a Flux neural network.

Notice that we are also able to define the neural network in a high-level fashion by only
specifying the number of neurons in each hidden layer and the activation
function. Meanwhile, `MLJFlux` is able to infer the input and output layer as well as use
a suitable default for the loss function and output activation given the classification
task. Notice as well that we did not need to manually implement a training or prediction
loop.

## Basic idea: "builders" for data-dependent architecture

As in the example above, any MLJFlux model has a `builder` hyperparameter, an object
encoding instructions for creating a neural network given the data that the model
eventually sees (e.g., the number of classes in a classification problem). While each MLJ
model has a simple default builder, users may need to define custom builders to get
optimal results (see [Defining Custom Builders](@ref) and this will require familiarity
with the [Flux API](https://fluxml.ai/Flux.jl/stable/) for defining a neural network
chain.


## Flux or MLJFlux?
[Flux](https://fluxml.ai/Flux.jl/stable/) is a deep learning framework in Julia that comes with everything you need to build deep learning models (i.e., GPU support, automatic differentiation, layers, activations, losses, optimizers, etc.). [MLJFlux](https://github.com/FluxML/MLJFlux.jl) wraps models built with Flux which provides a more high-level interface for building and training such models. More importantly, it empowers Flux models by extending their support to many common machine learning workflows that are possible via MLJ such as:

- **Estimating performance** of your model using a holdout set or other resampling strategy (e.g., cross-validation) as measured by one or more metrics (e.g., loss functions) that may not have been used in training

- **Optimizing hyper-parameters** such as a regularization parameter (e.g., dropout) or a width/height/nchannnels of convolution layer

- **Compose with other models** such as introducing data pre-processing steps (e.g., missing data imputation) into a pipeline. It might make sense to include non-deep learning models in this pipeline. Other kinds of model composition could include blending predictions of a deep learner with some other kind of model (as in “model stacking”). Models composed with MLJ can be also tuned as a single unit.

- **Controlling iteration** by adding an early stopping criterion based on an out-of-sample estimate of the loss, dynamically changing the learning rate (eg, cyclic learning rates), periodically save snapshots of the model, generate live plots of sample weights to judge training progress (as in tensor board)


- **Comparing** your model with a non-deep learning models

A comparable project, [FastAI](https://github.com/FluxML/FastAI.jl)/[FluxTraining](https://github.com/FluxML/FluxTraining.jl), also provides a high-level interface for interacting with Flux models and supports a set of features that may overlap with (but not include all of) those supported by MLJFlux.

Many of the features mentioned above are showcased in the workflow examples that you can access from the sidebar.
