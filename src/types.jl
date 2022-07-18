abstract type MLJFluxProbabilistic <: MLJModelInterface.Probabilistic end
abstract type MLJFluxDeterministic <: MLJModelInterface.Deterministic end

const MLJFluxModel = Union{MLJFluxProbabilistic,MLJFluxDeterministic}

for Model in [:NeuralNetworkClassifier, :ImageClassifier]

    ex = quote
        mutable struct $Model{B,F,O,L} <: MLJFluxProbabilistic
            builder::B
            finaliser::F
            optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
            loss::L        # can be called as in `loss(yhat, y)`
            epochs::Int    # number of epochs
            batch_size::Int  # size of a batch
            lambda::Float64  # regularization strength
            alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
            rng::Union{AbstractRNG,Int64}
            optimiser_changes_trigger_retraining::Bool
            acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
        end

        function $Model(; builder::B   = Short()
                        , finaliser::F = Flux.softmax
                        , optimiser::O = Flux.Optimise.Adam()
                        , loss::L      = Flux.crossentropy
                        , epochs       = 10
                        , batch_size   = 1
                        , lambda       = 0
                        , alpha        = 0
                        , rng          = Random.GLOBAL_RNG
                        , optimiser_changes_trigger_retraining = false
                        , acceleration = CPU1()
                        ) where {B,F,O,L}

            model = $Model{B,F,O,L}(builder
                                    , finaliser
                                    , optimiser
                                    , loss
                                    , epochs
                                    , batch_size
                                    , lambda
                                    , alpha
                                    , rng
                                    , optimiser_changes_trigger_retraining
                                    , acceleration
                                    )

            message = clean!(model)
            isempty(message) || @warn message

            return model
        end

    end
    eval(ex)

end

"""
$(MMI.doc_header(NeuralNetworkClassifier))

`NeuralNetworkClassifier` is for training a data-dependent Flux.jl neural network
for making probabilistic predictions of a `Multiclass` or `OrderedFactor` target,
given a table of `Continuous` features. Users provide a recipe for constructing
 the network, based on properties of the data that is encountered, by specifying
 an appropriate `builder`. See MLJFlux documentation for more on builders.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the column scitypes with `schema(X)`.
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Multiclass` or `OrderedFactor` with `n_out` classes;
  check the scitype with `scitype(y)`


# Hyper-parameters

- `builder=MLJFlux.Short()`: An MLJFlux builder that constructs a neural
   network. Possible `builders` include: `MLJFlux.Linear`, `MLJFlux.Short`,
   and `MLJFlux.MLP`. See MLJFlux documentation for examples of
   user-defined builders.
- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the updating of the weights of the network. For further reference, see either the examples or [the Flux optimiser documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a learning rate (the update rate of the optimizer), a good rule of thumb is to start out at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.crossentropy`: The loss function which the network will optimize. Should be a function which can be called in the form `loss(yhat, y)`. Possible loss functions are listed in [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a classification task, the most natural loss functions are:
    - `Flux.crossentropy`: Typically used as loss in multiclass classification, with labels in a 1-hot encoded format.
    - `Flux.logitcrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `softmax` and then calculating crossentropy.
    - `Flux.binarycrossentropy`: Typically used as loss in binary classification, with labels in a 1-hot encoded format.
    - `Flux.logitbinarycrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `sigmoid` and then calculating binary crossentropy.
    - `Flux.tversky_loss`: Used with imbalanced data to give more weight to false negatives.
    - `Flux.focal_loss`: Used with highly imbalanced data. Weights harder examples more than easier examples.
    - `Flux.binary_focal_loss`: Binary version of the above
    Currently MLJ measures are not supported as loss functions here.
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents one pass through the entirety of the training dataset.
- `batch_size::int=1`: the batch size to be used for training. the batch size represents
  the number of samples per update of the networks weights. typcally, batch size should be
  somewhere between 8 and 512. smaller batch sizes lead to noisier training loss curves,
  while larger batch sizes lead towards smoother training loss curves.
  In general, it is a good idea to pick one fairly large batch size (e.g. 32, 64, 128),
  and stick with it, and only tune the learning rate. In most examples, batch size is set
  in powers of twos, but this is fairly arbitrary.
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be any value  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a machine if the associated optimiser has changed. If true, the associated machine will retrain from scratch on `fit!`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For Training on GPU, use `CudaLibs()`. For training on GPU, use `CUDALibs()`.
- `finaliser=Flux.softmax`: The final activation function of the neural network. Defaults to `Flux.softmax`. For a classification task, `softmax` is used for multiclass, single label regression, `sigmoid` is used for either binary classification or multi label classification (when there are multiple possible labels for a given sample).


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above. Predictions are
  probabilistic but uncalibrated.
- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers,
   functions, and activations  which make up the neural network. This includes
   the final layer specified by `finaliser` (eg, `softmax`).


# Report

The fields of `report(mach)` are:

- `training_losses`: A vector of training losses (penalised if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we build a classification model using the Iris dataset.
```julia
using MLJ
using Flux
import RDatasets

using Random
Random.seed!(123)

```
This is a very basic example, using a default builder and no standardization.
For a more advanced illustration, see [`NeuralNetworkRegressor`](@ref) or [`ImageClassifier`](@ref). First, we can load the data:
```julia
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), rng=123);
NeuralNetworkClassifier = @load NeuralNetworkClassifier
clf = NeuralNetworkClassifier()
```
Next, we can train the model:
```julia
import Random.seed!; seed!(123)
mach = machine(clf, X, y)
fit!(mach)
```
We can train the model in an incremental fashion with the `optimizer_changes_trigger_retraining` flag set to false (which is by default). Here, we change the number of iterations and the learning rate of the optimiser:
```julia
clf.optimiser.eta = clf.optimiser.eta * 2
clf.epochs = clf.epochs + 5

# note that if the `optimizer_changes_trigger_retraining` flag was set to true
# the model would be completely retrained from scratch because the optimizer was
# updated
fit!(mach, verbosity=2);
```
We can inspect the mean training loss using the `cross_entropy` function:
```julia

training_loss = cross_entropy(predict(mach, X), y) |> mean

```
And we can access the Flux chain (model) using `fitted_params`:
```julia
chain = fitted_params(mach).chain
```
Finally, we can see how the out-of-sample performance changes over time, using the `learning_curve` function
```julia
r = range(clf, :epochs, lower=1, upper=200, scale=:log10)
curve = learning_curve(clf, X, y,
                     range=r,
                     resampling=Holdout(fraction_train=0.7),
                     measure=cross_entropy)
using Plots
plot(curve.parameter_values,
     curve.measurements,
     xlab=curve.parameter_name,
     xscale=curve.parameter_scale,
     ylab = "Cross Entropy")

```
See also
[`ImageClassifier`](@ref)
"""
NeuralNetworkClassifier

"""
$(MMI.doc_header(ImageClassifier))

`ImageClassifier` classifies images using a neural network adapted to the type
 of images provided (color or greyscale). Predictions are probabistic. Users
 provide a recipe for constructing the network, based on properties of the image
 encountered, by specifying an appropriate `builder`. See MLJFlux documentation
 for more on builders.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where
- `X`: is any `AbstractVector` of images with `ColorImage` or `GrayImage`
   scitype; check the scitype with `scitype(X)` and refer to ScientificTypes.jl
   documentation on coercing typical image formats into an appropriate type.
- `y`: is the target, which can be any `AbstractVector` whose element
   scitype is `Multiclass`; check the scitype with `scitype(y)`.


# Hyper-parameters

- `builder`: An MLJFlux builder that constructs the neural network.
   The fallback builds a depth-16 VGG architecture adapted to the image
   size and number of target classes, with no batch normalisation; see the
   Metalhead.jl documentation for details. See the example below for a
   user-specified builder.
- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the updating of the weights of the network. For further reference, see either the examples or [the Flux optimiser documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a learning rate (the update rate of the optimizer), a good rule of thumb is to start out at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.crossentropy`: The loss function which the network will optimize. Should be a function which can be called in the form `loss(yhat, y)`. Possible loss functions are listed in [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a classification task, the most natural loss functions are:
    - `Flux.crossentropy`: Typically used as loss in multiclass classification, with labels in a 1-hot encoded format.
    - `Flux.logitcrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `softmax` and then calculating crossentropy.
    - `Flux.binarycrossentropy`: Typically used as loss in binary classification, with labels in a 1-hot encoded format.
    - `Flux.logitbinarycrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `sigmoid` and then calculating binary crossentropy.
    - `Flux.tversky_loss`: Used with imbalanced data to give more weight to false negatives.
    - `Flux.focal_loss`: Used with highly imbalanced data. Weights harder examples more than easier examples.
    - `Flux.binary_focal_loss`: Binary version of the above
    Currently MLJ measures are not supported as loss functions here.
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size
  represents the number of samples per update of the networks weights. Batch
  sizes between 8 and 512 are typical. Increasing batch size can speed up
  training, especially on a GPU (`acceleration=CUDALibs()`).
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be any value  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a machine if the associated optimiser has changed. If true, the associated machine will retrain from scratch on `fit!`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For Training on GPU, use `CudaLibs()`. For training on GPU, use `CUDALibs()`.
- `finaliser=Flux.softmax`: The final activation function of the neural network,
    needed to convert outputs to probabilities (builders do not provide this).


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above. Predictions are
  probabilistic but uncalibrated.
- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers,
   functions, and activations  which make up the neural network. This includes
   the final layer specified by `finaliser` (eg, `softmax`).


# Report

The fields of `report(mach)` are:
- `training_losses`: A vector of training losses (penalised if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we use MLJ to classify the MNIST image dataset
```julia
using MLJ
using Flux
import MLJFlux
import MLJIteration # for `skip`

```
First we want to download the MNIST dataset, and unpack into images and labels
```julia
import MLDatasets: MNIST

images, labels = MNIST.traindata();
```
In MLJ, integers cannot be used for encoding categorical data, so we must coerce them into the `Multiclass` scitype:
```julia
labels = coerce(labels, Multiclass);
images = coerce(images, GrayImage);

images[1]
```
We start by defining a suitable `builder` object. This is a recipe
for building the neural network. Our builder will work for images of
any (constant) size, whether they be color or black and white (ie,
single or multi-channel).  The architecture always consists of six
alternating convolution and max-pool layers, and a final dense
layer; the filter size and the number of channels after each
convolution layer is customisable.
```julia
import MLJFlux

struct MyConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end

make2d(x::AbstractArray) = reshape(x, :, size(x)[end])

function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3
    mod(k, 2) == 1 || error("`filter_size` must be odd. ")
    p = div(k - 1, 2) # padding to preserve image size
    init = Flux.glorot_uniform(rng)
    front = Chain(
        Conv((k, k), n_channels => c1, pad=(p, p), relu, init=init),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad=(p, p), relu, init=init),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad=(p, p), relu, init=init),
        MaxPool((2 ,2)),
        make2d)
    d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
    return Chain(front, Dense(d, n_out, init=init))
end
```
It is important to note that in our `build` function, there is no final `softmax`. This is applied by default in all MLJFlux classifiers (override this using the `finaliser` hyperparameter). Now that we have our builder defined, we can define the actual model. If you have a GPU, you can substitute in `acceleration=CUDALibs()` below to  greatly speed up training.
```julia
ImageClassifier = @load ImageClassifier
clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
                      batch_size=50,
                      epochs=10,
                      rng=123)
```
You can add flux options such as `optimiser` and `loss` in the snippet above. Currently, `loss` must be a flux-compatible loss, and not an MLJ measure.
Next, we can bind the model with the data in a machine, and fit the first 500 or so images:
```julia
mach = machine(clf, images, labels);

fit!(mach, rows=1:500, verbosity=2);

report(mach)

chain = fitted_params(mach)

Flux.params(chain)[2]
```
We can tack on 20 more epochs by modifying the `epochs` field, and iteratively fit some more:
```julia
clf.epochs = clf.epochs + 20
fit!(mach, rows=1:500);
```
We can also make predictions and calculate an out-of-sample loss estimate:
```julia
predicted_labels = predict(mach, rows=501:1000);
cross_entropy(predicted_labels, labels[501:1000]) |> mean
```
The preceding `fit!`/`predict`/evaluate workflow can be alternatively executed as folllows:

```julia
evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)
```
See also
[`NeuralNetworkClassifier`](@ref)
"""
ImageClassifier

for Model in [:NeuralNetworkRegressor, :MultitargetNeuralNetworkRegressor]

    ex = quote
        mutable struct $Model{B,O,L} <: MLJFluxDeterministic
            builder::B
            optimiser::O  # mutable struct from Flux/src/optimise/optimisers.jl
            loss::L       # can be called as in `loss(yhat, y)`
            epochs::Int   # number of epochs
            batch_size::Int # size of a batch
            lambda::Float64 # regularization strength
            alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
            rng::Union{AbstractRNG,Integer}
            optimiser_changes_trigger_retraining::Bool
            acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
        end

        function $Model(; builder::B   = Linear()
                        , optimiser::O = Flux.Optimise.Adam()
                        , loss::L      = Flux.mse
                        , epochs       = 10
                        , batch_size   = 1
                        , lambda       = 0
                        , alpha        = 0
                        , rng          = Random.GLOBAL_RNG
                        , optimiser_changes_trigger_retraining=false
                        , acceleration  = CPU1()
                        ) where {B,O,L}

            model = $Model{B,O,L}(builder
                                  , optimiser
                                  , loss
                                  , epochs
                                  , batch_size
                                  , lambda
                                  , alpha
                                  , rng
                                  , optimiser_changes_trigger_retraining
                                  , acceleration)

            message = clean!(model)
            isempty(message) || @warn message

            return model
        end

    end
    eval(ex)

end


"""
$(MMI.doc_header(NeuralNetworkRegressor))

`NeuralNetworkRegressor` is for training a data-dependent Flux.jl neural
network to predict a `Continuous` target, given a table of
`Continuous` features. Users provide a recipe for constructing the
network, based on properties of the data that is encountered, by specifying
an appropriate `builder`. See MLJFlux documentation for more on builders.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the column scitypes with `schema(X)`.
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`


# Hyper-parameters

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs
   a neural network. Possible `builders` include: `MLJFlux.Linear`, `MLJFlux.Short`,
   and `MLJFlux.MLP`. See below for an example of a user-specified builder.
- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the updating
  of the weights of the network. For further reference, see either the examples or
  [the Flux optimiser documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/).
  To choose a learning rate (the update rate of the optimizer), a good rule of thumb is to
  start out at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.mse`: The loss function which the network will optimize. Should be a function
  which can be called in the form `loss(yhat, y)`.
  Possible loss functions are listed in [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/).
  For a regression task, the most natural loss functions are:
    - `Flux.mse`
    - `Flux.mae`
    - `Flux.msle`
    - `Flux.huber_loss`
    Currently MLJ measures are not supported as loss functions here.
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents
  one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size
  represents the number of samples per update of the networks weights. Batch
  sizes between 8 and 512 are typical. Increasing batch size can speed up
  training, especially on a GPU (`acceleration=CUDALibs()`).
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be any value
  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`.
  A value of 0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a machine if the associated optimiser has changed. If true, the associated machine will retrain from scratch on `fit!`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done.
For training on GPU, use `CudaLibs()`. For training on GPU, use `CUDALibs()`.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above. Predictions are
  deterministic.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers,
   functions, and activations  which make up the neural network. This includes
   the final layer specified by `finaliser` (eg, `softmax`).


# Report

The fields of `report(mach)` are:

- `training_losses`: A vector of training losses (penalised if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we build a regression model using the Boston house price dataset
```julia
  using MLJ
  using MLJFlux
  using Flux
```
First, we load in the data, with target `:MEDV`. We load in all features except `:CHAS`:
```julia
data = OpenML.load(531); # Loads from https://www.openml.org/d/531

y, X = unpack(data, ==(:MEDV), !=(:CHAS); rng=123);

scitype(y)
schema(X)
```
Since MLJFlux models do not handle ordered factors, we can treat `:RAD` as `Continuous`:
```julia
X = coerce(X, :RAD=>Continuous)
```
Lets also make a test set:
```julia
(X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);
```
Next, we can define a `builder`. In the following macro call, `n_in` is the number of expected input features, and rng is a RNG. `init` is the function used to generate the random initial weights of the network.
expected input features, and rng is a RNG. `init` is the function used to generate the random initial weights of the network.
random initial weights of the network.
```julia
builder = MLJFlux.@builder begin
  init=Flux.glorot_uniform(rng)
  Chain(Dense(n_in, 64, relu, init=init),
        Dense(64, 32, relu, init=init),
        Dense(32, 1, init=init))
end
```
Finally, we can define the model!
```julia
NeuralNetworkRegressor = @load NeuralNetworkRegressor
  model = NeuralNetworkRegressor(builder=builder,
                                 rng=123,
                                 epochs=20)
```
We will arrange for standardizaion of the the target by wrapping our model
 in `TransformedTargetModel`, and standardization of the features by
inserting the wrapped model in a pipeline:
```julia
pipe = Standardizer |> TransformedTargetModel(model, target=Standardizer)
```
If we fit with a high verbosity (>1), we will see the losses during training. We can also see the losses in the output of `report(mach)`
also see the losses in the output of `report(mach)`
```julia
mach = machine(pipe, X, y)
fit!(mach, verbosity=2)

# first element initial loss, 2:end per epoch training losses
report(mach).transformed_target_model_deterministic.training_losses

```

## Experimenting with learning rate

We can visually compare how the learning rate affects the predictions:
```julia
using Plots

rates = 10. .^ (-5:0)

foreach(rates) do η
  pipe.transformed_target_model_deterministic.model.optimiser.eta = η
  fit!(mach, force=true, verbosity=0)
  losses =
      report(mach).transformed_target_model_deterministic.model.training_losses[3:end]
  plot!(1:length(losses), losses, label=η)
end


pipe.transformed_target_model_deterministic.model.optimiser.eta = 0.0001

# CV estimate, based on `(X, y)`:
evaluate!(mach, resampling=CV(nfolds=5), measure=l2)

# loss for `(Xtest, test)`:
fit!(mach) # train on `(X, y)`
yhat = predict(mach, Xtest)
l2(yhat, ytest)  |> mean
```

For impementing stopping criterion and other iteration controls, refer to examples linked
from the MLJFlux documentation

See also
[`MultitargetNeuralNetworkRegressor`](@ref)
"""
NeuralNetworkRegressor

"""
$(MMI.doc_header(MultitargetNeuralNetworkRegressor))

`MultitargetNeuralNetworkRegressor` is for training a data-dependent Flux.jl
 neural network to predict a multivalued `Continuous` target, represented as a table,
 given a table of `Continuous` features. Users provide a recipe for constructing the
 network, based on properties of the data that is encountered, by specifying an
appropriate `builder`. See MLJFlux documentation for more on builders.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the column scitypes with `schema(X)`.
- `y`: is the target, which can be any table of output targets whose element
  scitype is `Continuous`; check the scitype with `schema(y)`


# Hyper-parameters

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs a neural
  network. Possible `builders` include: `Linear`, `Short`, and `MLP`. You can construct
  your own builder using the `@builder` macro, see examples for further information.
- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the
  updating of the weights of the network. For further reference, see either the examples
  or [the Flux optimiser
  documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a
  learning rate (the update rate of the optimizer), a good rule of thumb is to start out
  at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.mse`: The loss function which the network will optimize. Should be a
  function which can be called in the form `loss(yhat, y)`. Possible loss functions are
  listed in [the Flux loss function
  documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a regression task,
  the most natural loss functions are:
    - `Flux.mse`
    - `Flux.mae`
    - `Flux.msle`
    - `Flux.huber_loss`
    Currently MLJ measures are not supported as loss functions here.
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents
  one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size
  represents the number of samples per update of the networks weights. Batch
  sizes between 8 and 512 are typical. Increasing batch size can speed up
  training, especially on a GPU (`acceleration=CUDALibs()`).
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be
  any value  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of
  0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during
  training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a machine if the associated optimiser has changed. If true, the associated machine will retrain from scratch on `fit!`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done.
For Training on GPU, use `CudaLibs()`. For training on GPU, use `CUDALibs()`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above. Predictions are
  deterministic.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers,
   functions, and activations  which make up the neural network. This includes
   the final layer specified by `finaliser` (eg, `softmax`).


# Report

The fields of `report(mach)` are:

- `training_losses`: A vector of training losses (penalised if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we build a regression model using a toy dataset.
```julia
using MLJ
using MLJFlux
using Flux
using MLJBase: augment_X
```
First, we generate some data:
```julia
X = augment_X(randn(10000, 8), true);
θ = randn((9,2));
y = X * θ;
X = MLJ.table(X)
y = MLJ.table(y)

schema(y)
schema(X)
```
Lets also make a test set:
```julia
(X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);
```
Next, we can define a `builder`. In the following macro call, `n_in` is the number of expected input features, and rng is a RNG. `init` is the function used to generate the random initial weights of the network.
```julia
builder = MLJFlux.@builder begin
  init=Flux.glorot_uniform(rng)
  Chain(Dense(n_in, 64, relu, init=init),
        Dense(64, 32, relu, init=init),
        Dense(32, 1, init=init))
end
```
Finally, we can define the model!
```julia
MultitargetNeuralNetworkRegressor = @load MultitargetNeuralNetworkRegressor
model = MultitargetNeuralNetworkRegressor(builder=builder, rng=123, epochs=20)
```
We will arrange for standardizaion of the the target by wrapping our model
 in `TransformedTargetModel`, and standardization of the features by
inserting the wrapped model in a pipeline:
```julia
pipe = Standardizer |> TransformedTargetModel(model, target=Standardizer)
```
If we fit with a high verbosity (>1), we will see the losses during training. We can also see the losses in the output of `report(mach)`

```julia
mach = machine(pipe, X, y)
fit!(mach, verbosity=2)

# first element initial loss, 2:end per epoch training losses
report(mach).transformed_target_model_deterministic.training_losses

```

## Experimenting with learning rate

We can visually compare how the learning rate affects the predictions:
```julia
using Plots

rates = 10. .^ (-5:0)

foreach(rates) do η
  pipe.transformed_target_model_deterministic.model.optimiser.eta = η
  fit!(mach, force=true, verbosity=0)
  losses =
      report(mach).transformed_target_model_deterministic.model.training_losses[3:end]
  plot!(1:length(losses), losses, label=η)
end



pipe.transformed_target_model_deterministic.model.optimiser.eta = 0.0001

```

With the learning rate fixed, we can now compute a CV estimate of the performance (using
all data bound to `mach`) and compare this with performance on the test set:
```julia
# custom MLJ loss:
multi_loss(yhat, y) = l2(MLJ.matrix(yhat), MLJ.matrix(y)) |> mean

# CV estimate, based on `(X, y)`:
evaluate!(mach, resampling=CV(nfolds=5), measure=multi_loss)

# loss for `(Xtest, test)`:
fit!(mach)
yhat = predict(mach, Xtest)
multi_loss(yhat, y)
```

See also
[`NeuralNetworkRegressor`](@ref)
"""
MultitargetNeuralNetworkRegressor

const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}
