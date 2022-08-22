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



const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}


MMI.metadata_pkg.(
    (
        NeuralNetworkRegressor,
        MultitargetNeuralNetworkRegressor,
        NeuralNetworkClassifier,
        ImageClassifier,
    ),
    name="MLJFlux",
    uuid="094fc8d1-fd35-5302-93ea-dabda2abf845",
    url="https://github.com/alan-turing-institute/MLJFlux.jl",
    julia=true,
    license="MIT",
)


# # DOCSTRINGS

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

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is `Multiclass`
  or `OrderedFactor`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `builder=MLJFlux.Short()`: An MLJFlux builder that constructs a neural network. Possible
   `builders` include: `MLJFlux.Linear`, `MLJFlux.Short`, and `MLJFlux.MLP`. See
   MLJFlux.jl documentation for examples of user-defined builders. See also `finaliser`
   below.

- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the
  updating of the weights of the network. For further reference, see [the Flux optimiser
  documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a
  learning rate (the update rate of the optimizer), a good rule of thumb is to start out
  at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.

- `loss=Flux.crossentropy`: The loss function which the network will optimize. Should be a
  function which can be called in the form `loss(yhat, y)`.  Possible loss functions are
  listed in [the Flux loss function
  documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a classification
  task, the most natural loss functions are:

  - `Flux.crossentropy`: Standard multiclass classification loss, also known as the log
    loss.

  - `Flux.logitcrossentopy`: Mathematically equal to crossentropy, but numerically more
    stable than finalising the outputs with `softmax` and then calculating
    crossentropy. You will need to specify `finaliser=identity` to remove MLJFlux's
    default softmax finaliser, and understand that the output of `predict` is then
    unnormalized (no longer probabilistic).

  - `Flux.tversky_loss`: Used with imbalanced data to give more weight to false negatives.

  - `Flux.focal_loss`: Used with highly imbalanced data. Weights harder examples more than
    easier examples.

  Currently MLJ measures are not supported values of `loss`.

- `epochs::Int=10`: The duration of training, in epochs. Typically, one epoch represents
  one pass through the complete the training dataset.

- `batch_size::int=1`: the batch size to be used for training, representing the number of
  samples per update of the network weights. Typically, batch size is between 8 and
  512. Increassing batch size may accelerate training if `acceleration=CUDALibs()` and a
  GPU is available.

- `lambda::Float64=0`: The strength of the weight regularization penalty. Can be any value
  in the range `[0, ∞)`.

- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0
  represents L2 regularization, and a value of 1 represents L1 regularization.

- `rng::Union{AbstractRNG, Int64}`: The random number generator or seed used during
  training.

- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when re-fitting
  a machine if the associated optimiser has changed. If `true`, the associated machine
  will retrain from scratch on `fit!` call, otherwise it will not.

- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For
  Training on GPU, use `CUDALibs()`.

- `finaliser=Flux.softmax`: The final activation function of the neural network (applied
  after the network defined by `builder`). Defaults to `Flux.softmax`.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features `Xnew`, which
  should have the same scitype as `X` above. Predictions are probabilistic but uncalibrated.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions returned
  above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers,
   functions, and activations which make up the neural network. This includes
   the final layer specified by `finaliser` (eg, `softmax`).


# Report

The fields of `report(mach)` are:

- `training_losses`: A vector of training losses (penalised if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we build a classification model using the Iris dataset. This is a very
basic example, using a default builder and no standardization.  For a more advanced
illustration, see [`NeuralNetworkRegressor`](@ref) or [`ImageClassifier`](@ref), and
examples in the MLJFlux.jl documentation.

```julia
using MLJ
using Flux
import RDatasets
```

First, we can load the data:

```julia
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), rng=123); # a vector and a table
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
clf = NeuralNetworkClassifier()
```

Next, we can train the model:

```julia
mach = machine(clf, X, y)
fit!(mach)
```

We can train the model in an incremental fashion, altering the learning rate as we go,
provided `optimizer_changes_trigger_retraining` is `false` (the default). Here, we also
change the number of (total) iterations:

```julia
clf.optimiser.eta = clf.optimiser.eta * 2
clf.epochs = clf.epochs + 5

fit!(mach, verbosity=2) # trains 5 more epochs
```

We can inspect the mean training loss using the `cross_entropy` function:

```julia
training_loss = cross_entropy(predict(mach, X), y) |> mean
```

And we can access the Flux chain (model) using `fitted_params`:

```julia
chain = fitted_params(mach).chain
```

Finally, we can see how the out-of-sample performance changes over time, using MLJ's
`learning_curve` function:

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

See also [`ImageClassifier`](@ref).

"""
NeuralNetworkClassifier

"""
$(MMI.doc_header(ImageClassifier))

`ImageClassifier` classifies images using a neural network adapted to the type of images
provided (color or gray scale). Predictions are probabilistic. Users provide a recipe for
constructing the network, based on properties of the image encountered, by specifying an
appropriate `builder`. See MLJFlux documentation for more on builders.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any `AbstractVector` of images with `ColorImage` or `GrayImage` scitype; check
   the scitype with `scitype(X)` and refer to ScientificTypes.jl documentation on coercing
   typical image formats into an appropriate type.

- `y` is the target, which can be any `AbstractVector` whose element
   scitype is `Multiclass`; check the scitype with `scitype(y)`.

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `builder`: An MLJFlux builder that constructs the neural network.  The fallback builds a
   depth-16 VGG architecture adapted to the image size and number of target classes, with
   no batch normalization; see the Metalhead.jl documentation for details. See the example
   below for a user-specified builder. A convenience macro `@builder` is also
   available. See also `finaliser` below.

- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the
  updating of the weights of the network. For further reference, see [the Flux optimiser
  documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a
  learning rate (the update rate of the optimizer), a good rule of thumb is to start out
  at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.

- `loss=Flux.crossentropy`: The loss function which the network will optimize. Should be a
  function which can be called in the form `loss(yhat, y)`.  Possible loss functions are
  listed in [the Flux loss function
  documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a classification
  task, the most natural loss functions are:

  - `Flux.crossentropy`: Standard multiclass classification loss, also known as the log
    loss.

  - `Flux.logitcrossentopy`: Mathematically equal to crossentropy, but numerically more
    stable than finalising the outputs with `softmax` and then calculating
    crossentropy. You will need to specify `finaliser=identity` to remove MLJFlux's
    default softmax finaliser, and understand that the output of `predict` is then
    unnormalized (no longer probabilistic).

  - `Flux.tversky_loss`: Used with imbalanced data to give more weight to false negatives.

  - `Flux.focal_loss`: Used with highly imbalanced data. Weights harder examples more than
    easier examples.

  Currently MLJ measures are not supported values of `loss`.

- `epochs::Int=10`: The duration of training, in epochs. Typically, one epoch represents
  one pass through the complete the training dataset.

- `batch_size::int=1`: the batch size to be used for training, representing the number of
  samples per update of the network weights. Typically, batch size is between 8 and
  512. Increassing batch size may accelerate training if `acceleration=CUDALibs()` and a
  GPU is available.

- `lambda::Float64=0`: The strength of the weight regularization penalty. Can be any value
  in the range `[0, ∞)`.

- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0
  represents L2 regularization, and a value of 1 represents L1 regularization.

- `rng::Union{AbstractRNG, Int64}`: The random number generator or seed used during
  training.

- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when re-fitting
  a machine if the associated optimiser has changed. If `true`, the associated machine
  will retrain from scratch on `fit!` call, otherwise it will not.

- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For
  Training on GPU, use `CUDALibs()`.

- `finaliser=Flux.softmax`: The final activation function of the neural network (applied
  after the network defined by `builder`). Defaults to `Flux.softmax`.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features `Xnew`, which
  should have the same scitype as `X` above. Predictions are probabilistic but
  uncalibrated.

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

In this example we use MLJFlux and a custom builder to classify the MNIST image dataset.

```julia
using MLJ
using Flux
import MLJFlux
import MLJIteration # for `skip` control
```

First we want to download the MNIST dataset, and unpack into images and labels:

```julia
import MLDatasets: MNIST
data = MNIST(split=:train)
images, labels = data.features, data.targets
```

In MLJ, integers cannot be used for encoding categorical data, so we must coerce them into
the `Multiclass` scitype:

```julia
labels = coerce(labels, Multiclass);
```

Above `images` is a single array but MLJFlux requires the images to be a vector of
individual image arrays:

```
images = coerce(images, GrayImage);
images[1]
```

We start by defining a suitable `builder` object. This is a recipe for building the neural
network. Our builder will work for images of any (constant) size, whether they be color or
black and white (ie, single or multi-channel).  The architecture always consists of six
alternating convolution and max-pool layers, and a final dense layer; the filter size and
the number of channels after each convolution layer is customizable.

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

It is important to note that in our `build` function, there is no final `softmax`. This is
applied by default in all MLJFlux classifiers (override this using the `finaliser`
hyperparameter).

Now that our builder is defined, we can instantiate the actual MLJFlux model. If you have
a GPU, you can substitute in `acceleration=CUDALibs()` below to speed up training.

```julia
ImageClassifier = @load ImageClassifier pkg=MLJFlux
clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
                      batch_size=50,
                      epochs=10,
                      rng=123)
```

You can add Flux options such as `optimiser` and `loss` in the snippet above. Currently,
`loss` must be a flux-compatible loss, and not an MLJ measure.

Next, we can bind the model with the data in a machine, and train using the first 500
images:

```julia
mach = machine(clf, images, labels);
fit!(mach, rows=1:500, verbosity=2);
report(mach)
chain = fitted_params(mach)
Flux.params(chain)[2]
```

We can tack on 20 more epochs by modifying the `epochs` field, and iteratively fit some
more:

```julia
clf.epochs = clf.epochs + 20
fit!(mach, rows=1:500, verbosity=2);
```

We can also make predictions and calculate an out-of-sample loss estimate, using any MLJ
measure (loss/score):

```julia
predicted_labels = predict(mach, rows=501:1000);
cross_entropy(predicted_labels, labels[501:1000]) |> mean
```

The preceding `fit!`/`predict`/evaluate workflow can be alternatively executed as follows:

```julia
evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)
```

See also [`NeuralNetworkClassifier`](@ref).

"""
ImageClassifier

"""
$(MMI.doc_header(NeuralNetworkRegressor))

`NeuralNetworkRegressor` is for training a data-dependent Flux.jl neural network to
predict a `Continuous` target, given a table of `Continuous` features. Users provide a
recipe for constructing the network, based on properties of the data that is encountered,
by specifying an appropriate `builder`. See MLJFlux documentation for more on builders.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the column scitypes with `schema(X)`.
- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs a neural
   network. Possible `builders` include: `MLJFlux.Linear`, `MLJFlux.Short`, and
   `MLJFlux.MLP`. See MLJFlux documentation for more on builders, and the example below
   for using the `@builder` convenience macro.

- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the
  updating of the weights of the network. For further reference, see [the Flux optimiser
  documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a
  learning rate (the update rate of the optimizer), a good rule of thumb is to start out
  at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.

- `loss=Flux.mse`: The loss function which the network will optimize. Should be a function
  which can be called in the form `loss(yhat, y)`.  Possible loss functions are listed in
  [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/).
  For a regression task, natural loss functions are:

  - `Flux.mse`

  - `Flux.mae`

  - `Flux.msle`

  - `Flux.huber_loss`

  Currently MLJ measures are not supported as loss functions here.

- `epochs::Int=10`: The duration of training, in epochs. Typically, one epoch represents
  one pass through the complete the training dataset.

- `batch_size::int=1`: the batch size to be used for training, representing the number of
  samples per update of the network weights. Typically, batch size is between 8 and
  512. Increasing batch size may accelerate training if `acceleration=CUDALibs()` and a
  GPU is available.

- `lambda::Float64=0`: The strength of the weight regularization penalty. Can be any value
  in the range `[0, ∞)`.

- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0
  represents L2 regularization, and a value of 1 represents L1 regularization.

- `rng::Union{AbstractRNG, Int64}`: The random number generator or seed used during
  training.

- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when re-fitting
  a machine if the associated optimiser has changed. If `true`, the associated machine
  will retrain from scratch on `fit!` call, otherwise it will not.

- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For
  Training on GPU, use `CUDALibs()`.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features `Xnew`, which
  should have the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers, functions,
   and activations which make up the neural network.


# Report

The fields of `report(mach)` are:

- `training_losses`: A vector of training losses (penalized if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we build a regression model for the Boston house price dataset.

```julia
using MLJ
import MLJFlux
using Flux
```

First, we load in the data: The `:MEDV` column becomes the target vector `y`, and all
remaining columns go into a table `X`, with the exception of `:CHAS`:

```julia
data = OpenML.load(531); # Loads from https://www.openml.org/d/531
y, X = unpack(data, ==(:MEDV), !=(:CHAS); rng=123);

scitype(y)
schema(X)
```

Since MLJFlux models do not handle ordered factors, we'll treat `:RAD` as `Continuous`:

```julia
X = coerce(X, :RAD=>Continuous)
```

Splitting off a test set:

```julia
(X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);
```

Next, we can define a `builder`, making use of a convenience macro to do so.  In the
following `@builder` call, `n_in` is a proxy for the number input features (which will be
known at `fit!` time) and `rng` is a proxy for a RNG (which will be passed from the `rng`
field of `model` defined below). We also have the parameter `n_out` which is the number of
output features. As we are doing single target regression, the value passed will always be
`1`, but the builder we define will also work for [`MultitargetNeuralRegressor`](@ref).

```julia
builder = MLJFlux.@builder begin
    init=Flux.glorot_uniform(rng)
    Chain(
        Dense(n_in, 64, relu, init=init),
        Dense(64, 32, relu, init=init),
        Dense(32, n_out, init=init),
    )
end
```

Instantiating a model:

```julia
NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux
model = NeuralNetworkRegressor(
    builder=builder,
    rng=123,
    epochs=20
)
```

We arrange for standardization of the the target by wrapping our model in
`TransformedTargetModel`, and standardization of the features by inserting the wrapped
model in a pipeline:

```julia
pipe = Standardizer |> TransformedTargetModel(model, target=Standardizer)
```

If we fit with a high verbosity (>1), we will see the losses during training. We can also
see the losses in the output of `report(mach)`.

```julia
mach = machine(pipe, X, y)
fit!(mach, verbosity=2)

# first element initial loss, 2:end per epoch training losses
report(mach).transformed_target_model_deterministic.model.training_losses
```

## Experimenting with learning rate

We can visually compare how the learning rate affects the predictions:

```julia
using Plots

rates = rates = [5e-5, 1e-4, 0.005, 0.001, 0.05]
plt=plot()

foreach(rates) do η
  pipe.transformed_target_model_deterministic.model.optimiser.eta = η
  fit!(mach, force=true, verbosity=0)
  losses =
      report(mach).transformed_target_model_deterministic.model.training_losses[3:end]
  plot!(1:length(losses), losses, label=η)
end

plt

pipe.transformed_target_model_deterministic.model.optimiser.eta = 0.0001
```

With the learning rate fixed, we compute a CV estimate of the performance (using
all data bound to `mach`) and compare this with performance on the test set:

```julia
# CV estimate, based on `(X, y)`:
evaluate!(mach, resampling=CV(nfolds=5), measure=l2)

# loss for `(Xtest, test)`:
fit!(mach) # train on `(X, y)`
yhat = predict(mach, Xtest)
l2(yhat, ytest)  |> mean
```

These losses, for the pipeline model, refer to the target on the original, unstandardized,
scale.

For implementing stopping criterion and other iteration controls, refer to examples linked
from the MLJFlux documentation.

See also
[`MultitargetNeuralNetworkRegressor`](@ref)
"""
NeuralNetworkRegressor

"""
$(MMI.doc_header(MultitargetNeuralNetworkRegressor))

`MultitargetNeuralNetworkRegressor` is for training a data-dependent Flux.jl neural
network to predict a multi-valued `Continuous` target, represented as a table, given a
table of `Continuous` features. Users provide a recipe for constructing the network, based
on properties of the data that is encountered, by specifying an appropriate `builder`. See
MLJFlux documentation for more on builders.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of output targets whose element scitype is
  `Continuous`; check column scitypes with `schema(y)`.


# Hyper-parameters

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs a neural
  network. Possible `builders` include: `Linear`, `Short`, and `MLP`. See MLJFlux
  documentation for more on builders, and the example below for using the `@builder`
  convenience macro.

- `optimiser::Flux.Adam()`: A `Flux.Optimise` optimiser. The optimiser performs the
  updating of the weights of the network. For further reference, see [the Flux optimiser
  documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a
  learning rate (the update rate of the optimizer), a good rule of thumb is to start out
  at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.

- `loss=Flux.mse`: The loss function which the network will optimize. Should be a function
  which can be called in the form `loss(yhat, y)`.  Possible loss functions are listed in
  [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/).
  For a regression task, natural loss functions are:

  - `Flux.mse`

  - `Flux.mae`

  - `Flux.msle`

  - `Flux.huber_loss`

  Currently MLJ measures are not supported as loss functions here.

- `epochs::Int=10`: The duration of training, in epochs. Typically, one epoch represents
  one pass through the complete the training dataset.

- `batch_size::int=1`: the batch size to be used for training, representing the number of
  samples per update of the network weights. Typically, batch size is between 8 and
  512. Increassing batch size may accelerate training if `acceleration=CUDALibs()` and a
  GPU is available.

- `lambda::Float64=0`: The strength of the weight regularization penalty. Can be any value
  in the range `[0, ∞)`.

- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0
  represents L2 regularization, and a value of 1 represents L1 regularization.

- `rng::Union{AbstractRNG, Int64}`: The random number generator or seed used during
  training.

- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when re-fitting
  a machine if the associated optimiser has changed. If `true`, the associated machine
  will retrain from scratch on `fit!` call, otherwise it will not.

- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For
  Training on GPU, use `CUDALibs()`.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above. Predictions are
  deterministic.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain" (Flux.jl model), namely the series of layers,
   functions, and activations  which make up the neural network.


# Report

The fields of `report(mach)` are:

- `training_losses`: A vector of training losses (penalised if `lambda != 0`) in
   historical order, of length `epochs + 1`.  The first element is the pre-training loss.

# Examples

In this example we apply a multi-target regression model to synthetic data:

```julia
using MLJ
import MLJFlux
using Flux
```

First, we generate some synthetic data (needs MLJBase 0.20.16 or higher):

```julia
X, y = make_regression(100, 9; n_targets = 2) # both tables
schema(y)
schema(X)
```

Splitting off a test set:

```julia
(X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);
```

Next, we can define a `builder`, making use of a convenience macro to do so.  In the
following `@builder` call, `n_in` is a proxy for the number input features and `n_out` the
number of target variables (both known at `fit!` time), while `rng` is a proxy for a RNG
(which will be passed from the `rng` field of `model` defined below).

```julia
builder = MLJFlux.@builder begin
    init=Flux.glorot_uniform(rng)
    Chain(
        Dense(n_in, 64, relu, init=init),
        Dense(64, 32, relu, init=init),
        Dense(32, n_out, init=init),
    )
end
```

Instantiating the regression model:

```julia
MultitargetNeuralNetworkRegressor = @load MultitargetNeuralNetworkRegressor
model = MultitargetNeuralNetworkRegressor(builder=builder, rng=123, epochs=20)
```

We will arrange for standardization of the the target by wrapping our model in
 `TransformedTargetModel`, and standardization of the features by inserting the wrapped
 model in a pipeline:

```julia
pipe = Standardizer |> TransformedTargetModel(model, target=Standardizer)
```

If we fit with a high verbosity (>1), we will see the losses during training. We can also
see the losses in the output of `report(mach)`

```julia
mach = machine(pipe, X, y)
fit!(mach, verbosity=2)

# first element initial loss, 2:end per epoch training losses
report(mach).transformed_target_model_deterministic.model.training_losses
```

For experimenting with learning rate, see the [`NeuralNetworkRegressor`](@ref) example.

```
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
fit!(mach) # trains on all data `(X, y)`
yhat = predict(mach, Xtest)
multi_loss(yhat, ytest)
```

See also
[`NeuralNetworkRegressor`](@ref)
"""
MultitargetNeuralNetworkRegressor
