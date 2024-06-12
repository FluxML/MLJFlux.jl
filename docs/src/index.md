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
You only need `Flux` if you need to build a custom architecture, or experiment with different loss or activation functions. Since MLJFlux 0.5, you must use optimisers from Optimisers.jl, as native Flux.jl optimisers are no longer supported. 

## Quick Start

For the following demo, you will need to additionally run `Pkg.add("RDatasets")`.

```@example
using MLJ, Flux, MLJFlux
import RDatasets
import Optimisers

# 1. Load Data
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);

# 2. Load and instantiate model
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg="MLJFlux"
clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=100, 
    acceleration=CUDALibs()         # For GPU support
    )

# 3. Wrap it in a machine 
mach = machine(clf, X, y)

# 4. Evaluate the model
cv=CV(nfolds=5)
evaluate!(mach, resampling=cv, measure=accuracy) 
```
As you can see we are able to use MLJ meta-functionality (i.e., cross validation) with a Flux deep learning model. All arguments provided have defaults.

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
optimal results, and this will require familiarity with the [Flux
API](https://fluxml.ai/Flux.jl/stable/) for defining a neural network chain.


## Flux or MLJFlux?
[Flux](https://fluxml.ai/Flux.jl/stable/) is a deep learning framework in Julia that comes with everything you need to build deep learning models (i.e., GPU support, automatic differentiation, layers, activations, losses, optimizers, etc.). [MLJFlux](https://github.com/FluxML/MLJFlux.jl) wraps models built with Flux which provides a more high-level interface for building and training such models. More importantly, it empowers Flux models by extending their support to many common machine learning workflows that are possible via MLJ such as:

- **Estimating performance** of your model using a holdout set or other resampling strategy (e.g., cross-validation) as measured by one or more metrics (e.g., loss functions) that may not have been used in training

- **Optimizing hyper-parameters** such as a regularization parameter (e.g., dropout) or a width/height/nchannnels of convolution layer

- **Compose with other models** such as introducing data pre-processing steps (e.g., missing data imputation) into a pipeline. It might make sense to include non-deep learning models in this pipeline. Other kinds of model composition could include blending predictions of a deep learner with some other kind of model (as in “model stacking”). Models composed with MLJ can be also tuned as a single unit.

- **Controlling iteration** by adding an early stopping criterion based on an out-of-sample estimate of the loss, dynamically changing the learning rate (eg, cyclic learning rates), periodically save snapshots of the model, generate live plots of sample weights to judge training progress (as in tensor board)


- **Comparing** your model with a non-deep learning models

A comparable project, [FastAI](https://github.com/FluxML/FastAI.jl)/[FluxTraining](https://github.com/FluxML/FluxTraining.jl), also provides a high-level interface for interacting with Flux models and supports a set of features that may overlap with (but not include all of) those supported by MLJFlux.

Many of the features mentioned above are showcased in the workflow examples that you can access from the sidebar.
