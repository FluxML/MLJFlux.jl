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
                        , optimiser::O = Flux.Optimise.ADAM()
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

`NeuralNetworkClassifier`: a neural network model for making probabilistic predictions
of a Multiclass or OrderedFactor target, given a table of Continuous features. )
 TODO:

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Multiclass` or `OrderedFactor` with `n_out` classes;
  check the scitype with `scitype(y)`


# Hyper-parameters

- `builder=MLJFlux.Short()`: An MLJFlux builder that constructs a neural network. Possible `builders` include: `Linear`, `Short`, and `MLP`. You can construct your own builder using the `@builder` macro, see examples for further information.
- `optimiser::Flux.ADAM()`: A `Flux.Optimise` optimiser. The optimiser performs the updating of the weights of the network. For further reference, see either the examples or [the Flux optimiser documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a learning rate (the update rate of the optimizer), a good rule of thumb is to start out at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.crossentropy`: The loss function which the network will optimize. Should be a function which can be called in the form `loss(yhat, y)`. Possible loss functions are listed in [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a classification task, the most natural loss functions are:
    - `Flux.crossentropy`: Typically used as loss in multiclass classification, with labels in a 1-hot encoded format.
    - `Flux.logitcrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `softmax` and then calculating crossentropy.
    - `Flux.binarycrossentropy`: Typically used as loss in binary classification, with labels in a 1-hot encoded format.
    - `Flux.logitbinarycrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `sigmoid` and then calculating binary crossentropy.
    - `Flux.tversky_loss`: Used with imbalanced data to give more weight to false negatives.
    - `Flux.focal_loss`: Used with highly imbalanced data. Weights harder examples more than easier examples.
    - `Flux.binary_focal_loss`: Binary version of the above
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size represents the number of samples per update of the networks weights. Typcally, batch size should be somewhere between 8 and 512. Smaller batch sizes lead to noisier training loss curves, while larger batch sizes lead towards smoother training loss curves. In general, it is a good idea to pick one fairly large batch size (e.g. 32, 64, 128), and stick with it, and only tune the learning rate. In most literature, batch size is set in powers of twos, but this is fairly arbitrary.
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be any value  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a machine if the associated optimiser has changed. If true, the associated machine will retrain from scratch on `fit`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For Training on GPU, use `CudaLibs()`, otherwise defaults to `CPU`()`.
- `finaliser=Flux.softmax`: The final activation function of the neural network. Defaults to `Flux.softmax`. For a classification task, `softmax` is used for multiclass, single label regression, `sigmoid` is used for either binary classification or multi label classification (when there are multiple possible labels for a given sample).


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above. Predictions are
  probabilistic.
- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain", or series of layers, functions, and activations which make up the neural network.


# Report

The fields of `report(mach)` are:

- `training_losses`: The history of training losses, a vector containing the history of all the losses during training. The first element of the vector is the initial penalized loss. After the first element, the nth element corresponds to the loss of epoch n-1.

# Examples

In this example we build a classification model using the Iris dataset.
```julia
using MLJ
using Flux
import RDatasets

using Random
Random.seed!(123)

MLJ.color_off()

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));
```
This is a very basic example, using a default builder and no standardization.
For a more advance illustration, see [`NeuralNetworkRegressor`](@ref) or [`ImageClassifier`](@ref). First, we can load the data:
```julia
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
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

# note that if the optimizer_changes_trigger_retraining flag was set to true
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

savefig("iris_history.png")
```
See also
[`ImageClassifier`](@ref)
"""
NeuralNetworkClassifier

"""
$(MMI.doc_header(ImageClassifier))

`ImageClassifier`: A neural network model for making probabilistic
"predictions of a `GrayImage` target, given a table of `Continuous` features.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
mach = machine(model, X, y)
Where
- `X`: is any `AbstractVector` of input features (eg, a `DataFrame`) whose items
  are of scitype `GrayImage`; check the scitype with `scitype(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Multiclass` or `OrderedFactor` with `n_out` classes;
  check the scitype with `scitype(y)`


# Hyper-parameters

- `builder=MLJFlux.Short()`: An MLJFlux builder that constructs a neural network. Possible `builders` include: `Linear`, `Short`, and `MLP`. You can construct your own builder using the `@builder` macro, see examples for further information.
- `optimiser::Flux.ADAM()`: A `Flux.Optimise` optimiser. The optimiser performs the updating of the weights of the network. For further reference, see either the examples or [the Flux optimiser documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a learning rate (the update rate of the optimizer), a good rule of thumb is to start out at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.crossentropy`: The loss function which the network will optimize. Should be a function which can be called in the form `loss(yhat, y)`. Possible loss functions are listed in [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a classification task, the most natural loss functions are:
    - `Flux.crossentropy`: Typically used as loss in multiclass classification, with labels in a 1-hot encoded format.
    - `Flux.logitcrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `softmax` and then calculating crossentropy.
    - `Flux.binarycrossentropy`: Typically used as loss in binary classification, with labels in a 1-hot encoded format.
    - `Flux.logitbinarycrossentopy`: Mathematically equal to crossentropy, but computationally more numerically stable than finalising the outputs with `sigmoid` and then calculating binary crossentropy.
    - `Flux.tversky_loss`: Used with imbalanced data to give more weight to false negatives.
    - `Flux.focal_loss`: Used with highly imbalanced data. Weights harder examples more than easier examples.
    - `Flux.binary_focal_loss`: Binary version of the above
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size represents the number of samples per update of the networks weights. Typcally, batch size should be somewhere between 8 and 512. Smaller batch sizes lead to noisier training loss curves, while larger batch sizes lead towards smoother training loss curves. In general, it is a good idea to pick one fairly large batch size (e.g. 32, 64, 128), and stick with it, and only tune the learning rate. In most literature, batch size is set in powers of twos, but this is fairly arbitrary.
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be any value  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of 0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a machine if the associated optimiser has changed. If true, the associated machine will retrain from scratch on `fit`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done. For Training on GPU, use `CudaLibs()`, otherwise defaults to `CPU`()`.
- `finaliser=Flux.softmax`: The final activation function of the neural network. Defaults to `Flux.softmax`. For a regression task, reasonable alternatives include `Flux.sigmoid` and the identity function (otherwise known as "linear activation").


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above. Predictions are
  probabilistic.
- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.


# Fitted parameters

The fields of `fitted_params(mach)` are:
- `chain`: The trained "chain", or series of layers, functions, and activations which make up the neural network.


# Report

The fields of `report(mach)` are:
- `training_losses`: The history of training losses, a vector containing the history of all the losses during training. The first element of the vector is the initial penalized loss. After the first element, the nth element corresponds to the loss of epoch n-1.

# Examples

In this example we use MLJ to classify the MNIST image dataset
```julia
using MLJ
using Flux
import MLJFlux
import MLJIteration # for `skip`

MLJ.color_off()

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));
```
First we want to download the MNIST dataset, and unpack into images and labels
```julia
import MLDatasets: MNIST

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
images, labels = MNIST.traindata();
```
In MLJ, integers cannot be used for encoding categorical data, so we must coerce them into the `Multiclass` [scientific type](https://juliaai.github.io/ScientificTypes.jl/dev/). For more in this, see [Working with Categorical Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/):
```julia
labels = coerce(labels, Multiclass);
images = coerce(images, GrayImage);

# Checking scientific types:

@assert scitype(images) <: AbstractVector{<:Image}
@assert scitype(labels) <: AbstractVector{<:Finite}

images[1]
```
For general instructions on coercing image data, see [type coercion for image data](https://alan-turing-institute.github.io/ScientificTypes.jl/dev/%23Type-coercion-for-image-data-1)
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
It is important to note that in our `build` function, there is no final softmax. This is applie by default in all MLJFlux classifiers, using the `finaliser` hyperparameter of the classifier. Now that we have our builder defined, we can define the actual moel. If you have a GPU, you can substitute in `acceleration=CudaLibs()` below. Note that in the case of convolutions, this will **greatly** increase the speed of training.
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
We can also make predictions and calculate an out-of-sample loss estimate, in two ways!
```julia
predicted_labels = predict(mach, rows=501:1000);
cross_entropy(predicted_labels, labels[501:1000]) |> mean
# alternative one liner!
evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)
```

## Wrapping in iteration controls

Any iterative MLJFlux model can be wrapped in **iteration controls**, as we demonstrate next. For more on MLJ's `IteratedModel` wrapper, see the [MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/).
The "self-iterating" classifier (`iterated_clf` below) is for iterating the image classifier defined above until a stopping criterion is hit. We use the following stopping criterion:
- `Patience(3)`: 3 consecutive increases in the loss
- `InvalidValue()`: an out-of-sample loss or a training loss that is `NaN` or `±Inf`
- `TimeLimit(t=5/60)`: training time has exceeded 5 minutes.
We can specify how often these checks (and other controls) are applied using the `Step` control. Additionally, we can define controls to
- save a snapshot of the machine every N control cycles (`save_control`)
- record traces of the out-of-sample loss and training losses for plotting (`WithLossDo`)
- record mean value traces of each Flux parameter for plotting (`Callback`)
And other controls. For a full list, see [the documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/%23Controls-provided).
First, we define some helper functions and some empty vectors to store traces:
```julia
make2d(x::AbstractArray) = reshape(x, :, size(x)[end])
make1d(x::AbstractArray) = reshape(x, length(x));

# to extract the flux parameters from a machine
parameters(mach) = make1d.(Flux.params(fitted_params(mach)));

# trace storage
losses = []
training_losses = []
parameter_means = Float32[];
epochs = []

# to update traces
update_loss(loss) = push!(losses, loss)
update_training_loss(losses) = push!(training_losses, losses[end])
update_means(mach) = append!(parameter_means, mean.(parameters(mach)));
update_epochs(epoch) = push!(epochs, epoch)
```
Next, we can define our controls! We store them in a simple vector:
```julia
save_control =
    MLJIteration.skip(Save(joinpath(DIR, "mnist.jlso")), predicate=3)

controls=[Step(2),
          Patience(3),
          InvalidValue(),
          TimeLimit(5/60),
          save_control,
          WithLossDo(),
          WithLossDo(update_loss),
          WithTrainingLossesDo(update_training_loss),
          Callback(update_means),
          WithIterationsDo(update_epochs)
```
Once the controls are defined, we can instantiate and fit  our "self-iterating" classifier:
```julia
iterated_clf = IteratedModel(model=clf,
                       controls=controls,
                       resampling=Holdout(fraction_train=0.7),
  measure=log_loss)

mach = machine(iterated_clf, images, labels);
fit!(mach, rows=1:500);
```
Next we can compare the training and out-of-sample losses, as well as view the evolution of the weights:
```julia
plot(epochs, losses,
     xlab = "epoch",
     ylab = "root squared error",
     label="out-of-sample")
plot!(epochs, training_losses, label="training")

savefig(joinpath(DIR, "loss.png"))

n_epochs =  length(losses)
n_parameters = div(length(parameter_means), n_epochs)
parameter_means2 = reshape(copy(parameter_means), n_parameters, n_epochs)'
plot(epochs, parameter_means2,
     title="Flux parameter mean weights",
     xlab = "epoch")
# **Note.** The the higher the number, the deeper the chain parameter.
savefig(joinpath(DIR, "weights.png"))
```
Since we saved our model every few epochs, we can retrieve the snapshots so we can make predictions!
```julia
mach2 = machine(joinpath(DIR, "mnist3.jlso"))
predict_mode(mach2, images[501:503])
```

## Resuming training

If we change `iterated_clf.controls` or `clf.epochs`, we can resume training from where it left off. This is very useful for long-running training sessions, where you may be interrupted by for example a bad connection or computer hibernation.
```julia
iterated_clf.controls[2] = Patience(4)
fit!(mach, rows=1:500)

plot(epochs, losses,
     xlab = "epoch",
     ylab = "root squared error",
     label="out-of-sample")
plot!(epochs, training_losses, label="training")
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
                        , optimiser::O = Flux.Optimise.ADAM()
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

`NeuralNetworkRegressor`: A neural network model for making deterministic
predictions of a `Continuous` target, given a table of `Continuous` features.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`


# Hyper-parameters

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs a neural network.
  Possible `builders` include: `Linear`, `Short`, and `MLP`. You can construct your own builder
  using the `@builder` macro, see examples for further information.
- `optimiser::Flux.ADAM()`: A `Flux.Optimise` optimiser. The optimiser performs the updating
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
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents
  one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size represents
  the number of samples per update of the networks weights. Typcally, batch size should be
  somewhere between 8 and 512. Smaller batch sizes lead to noisier training loss curves,
  while larger batch sizes lead towards smoother training loss curves.
  In general, it is a good idea to pick one fairly large batch size (e.g. 32, 64, 128),
  and stick with it, and only tune the learning rate. In most examples, batch size is set
  in powers of twos, but this is fairly arbitrary.
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be any value
  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`.
  A value of 0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting a
  machine if the associated optimiser has changed. If true, the associated machine will
  retrain from scratch on `fit`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done.
  For training on GPU, use `CudaLibs()`, otherwise defaults to `CPU`()`.
- `finaliser=Flux.softmax`: The final activation function of the neural network.
  Defaults to `Flux.softmax`. For a regression task, reasonable alternatives include
  `Flux.sigmoid` and the identity function (otherwise known as "linear activation").


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above. Predictions are
  deterministic.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain", or series of layers, functions, and activations which
  make up the neural network.


# Report

The fields of `report(mach)` are:

- `training_losses`: The history of training losses, a vector containing the history of all the losses during training. The first element of the vector is the initial penalized loss. After the first element, the nth element corresponds to the loss of epoch n-1.
  all the losses during training. The first element of the vector is the initial penalized loss. After the first element, the nth element corresponds to the loss of epoch n-1.
  penalized loss. After the first element, the nth element corresponds to the loss of epoch n-1.
  epoch n-1.
# Examples

In this example we build a regression model using the Boston house price dataset
```julia
  using MLJ
  using MLJFlux
  using Flux
  using Plots
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
For our neural network, since different features likely have different scales, if we do not standardize the network may be implicitly biased towards features with higher magnitudes, or may have [saturated neurons](https://www.informit.com/articles/article.aspx%3fp=3131594&seqNum=2)  and not train well. Therefore, standardization is key!
not standardize the network may be implicitly biased towards features with higher magnitudes, or may have [saturated neurons](https://www.informit.com/articles/article.aspx%3fp=3131594&seqNum=2)  and not train well. Therefore, standardization is key!
magnitudes, or may have [saturated neurons](https://www.informit.com/articles/article.aspx%3fp=3131594&seqNum=2)  and not train well. Therefore, standardization is key!
neurons](https://www.informit.com/articles/article.aspx%3fp=3131594&seqNum=2)  and not train well. Therefore, standardization is key!
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
plt = plot()

rates = 10. .^ (-5:0)

foreach(rates) do η
  pipe.transformed_target_model_deterministic.model.optimiser.eta = η
  fit!(mach, force=true, verbosity=0)
  losses =
      report(mach).transformed_target_model_deterministic.model.training_losses[3:end]
  plot!(1:length(losses), losses, label=η)
end
plt #!md

savefig(joinpath("assets", "learning_rate.png"))

pipe.transformed_target_model_deterministic.model.optimiser.eta = 0.0001
```

## Using Iteration Controls

We can also wrap the model with MLJ Iteration controls. Suppose we want a model that trains until the out of sample loss does not improve for 6 epochs. We can use the `NumberSinceBest(6)` stopping criterion. We can also add some extra stopping criterion, `InvalidValue` and `Timelimit(1/60)`, as well as some controls to print traces of the losses. First we can define some methods to initialize or clear the traces as well as updte the traces.
trains until the out of sample loss does not improve for 6 epochs. We can use the `NumberSinceBest(6)` stopping criterion. We can also add some extra stopping criterion, `InvalidValue` and `Timelimit(1/60)`, as well as some controls to print traces of the losses. First we can define some methods to initialize or clear the traces as well as update the traces.
`NumberSinceBest(6)` stopping criterion. We can also add some extra stopping criterion, `InvalidValue` and `Timelimit(1/60)`, as well as some controls to print traces of the losses. First we can define some methods to initialize or clear the traces as well as update the traces.
```julia
# For initializing or clearing the traces:

clear() = begin
  global losses = []
  global training_losses = []
  global epochs = []
  return nothing
end

  # And to update the traces:

update_loss(loss) = push!(losses, loss)
update_training_loss(report) =
  push!(training_losses,
        report.transformed_target_model_deterministic.model.training_losses[end])
update_epochs(epoch) = push!(epochs, epoch)
```
For further reference of controls, see [the documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/%23Controls-provided). To apply the controls, we simply stack them in a vector and then make an `IteratedModel`:
```julia
controls=[Step(1),
        NumberSinceBest(6),
        InvalidValue(),
        TimeLimit(1/60),
        WithLossDo(update_loss),
        WithReportDo(update_training_loss),
WithIterationsDo(update_epochs)]


iterated_pipe =
  IteratedModel(model=pipe,
                controls=controls,
                resampling=Holdout(fraction_train=0.8),
                measure = l2)
```
Next, we can clear the traces, fit the model, and plot the traces:
```julia
clear()
mach = machine(iterated_pipe, X, y)
fit!(mach)

plot(epochs, losses,
   xlab = "epoch",
   ylab = "mean sum of squares error",
   label="out-of-sample",
   legend = :topleft);
scatter!(twinx(), epochs, training_losses, label="training", color=:red) #!md

savefig(joinpath("assets", "loss.png"))
```

### Brief note on iterated models

Training an `IteratedModel` means holding out some data (80% in this case) so an
out-of-sample loss can be tracked and used in the specified stopping criterion,
`NumberSinceBest(4)`. However, once the stop is triggered, the model wrapped by
`IteratedModel` (our pipeline model) is retrained on all data for the same number of
iterations. Calling `predict(mach, Xnew)` on new data uses the updated learned
parameters.

## Evaluating Iterated Models

We can evaluate our model with the `evaluate!` function:
```julia
e = evaluate!(mach,
             resampling=CV(nfolds=8),
             measures=[l1, l2])

using Measurements
l1_loss = e.measurement[1] ± std(e.per_fold[1])/sqrt(7)
@show l1_loss
```
We take this estimate of the uncertainty of the generalization error with a [grain of
salt](https://direct.mit.edu/neco/article-abstract/10/7/1895/6224/Approximate-Statistical-Tests-for-Comparing)).

## Comparison with other models on the test set

Although we cannot assign them statistical significance, here are comparisons, on the
untouched test set, of the eror of our self-iterating neural network regressor with a
couple of other models trained on the same data (using default hyperparameters):
```julia
function performance(model)
   mach = machine(model, X, y) |> fit!
   yhat = predict(mach, Xtest)
   l1(yhat, ytest) |> mean
end
performance(iterated_pipe)

three_models = [(@load EvoTreeRegressor)(), # tree boosting model
               (@load LinearRegressor pkg=MLJLinearModels)(),
               iterated_pipe]

errs = performance.(three_models)

(models=MLJ.name.(three_models), mean_square_errors=errs) |> pretty
```

See also
[`MultitargetNeuralNetworkRegressor`](@ref)
"""
NeuralNetworkRegressor

"""
$(MMI.doc_header(MultitargetNeuralNetworkRegressor))

`MultitargetNeuralNetworkRegressor`: A neural network model for making deterministic
predictions of a `Continuous` multi-target, presented as a table, given a table of
`Continuous` features.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`
- `y`: is the target, which can be any table of output targets whose element
  scitype is `Continuous`; check the scitype with `schema(y)`


# Hyper-parameters

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs a neural
  network. Possible `builders` include: `Linear`, `Short`, and `MLP`. You can construct
  your own builder using the `@builder` macro, see examples for further information.
- `optimiser::Flux.ADAM()`: A `Flux.Optimise` optimiser. The optimiser performs the
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
- `epochs::Int=10`: The number of epochs to train for. Typically, one epoch represents
  one pass through the entirety of the training dataset.
- `batch_size::Int=1`: The batch size to be used for training. The batch size represents
  the number of samples per update of the networks weights. Typcally, batch size should be
  somewhere between 8 and 512. Smaller batch sizes lead to noisier training loss curves,
  while larger batch sizes lead towards smoother training loss curves. In general, it is a
  good idea to pick one fairly large batch size (e.g. 32, 64, 128), and stick with it, and
  only tune the learning rate. In most literature, batch size is set in powers of twos,
  but this is fairly arbitrary.
- `lambda::Float64=0`: The stregth of the regularization used during training. Can be
  any value  in the range `[0, ∞)`.
- `alpha::Float64=0`: The L2/L1 mix of regularization, in the range `[0, 1]`. A value of
  0 represents L2 regularization, and a value of 1 represents L1 regularization.
- `rng::Union{AbstractRNG, Int64}`: The random number generator/seed used during
  training.
- `optimizer_changes_trigger_retraining::Bool=false`: Defines what happens when fitting
  a machine if the associated optimiser has changed. If true, the associated machine will
  retrain from scratch on `fit`, otherwise it will not.
- `acceleration::AbstractResource=CPU1()`: Defines on what hardware training is done.
  For Training on GPU, use `CudaLibs()`, otherwise defaults to `CPU`()`.
- `finaliser=Flux.softmax`: The final activation function of the neural network.
Defaults to `Flux.softmax`. For a regression task, reasonable alternatives include
`Flux.sigmoid` and the identity function (otherwise known as "linear activation").

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above. Predictions are
  deterministic.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `chain`: The trained "chain", or series of layers, functions, and activations which
  make up the neural network.


# Report

The fields of `report(mach)` are:

- `training_losses`: The history of training losses, a vector containing the history of
  all the losses during training. The first element of the vector is the initial
  penalized loss. After the first element, the nth element corresponds to the loss of
  epoch n-1.

# Examples

In this example we build a regression model using a toy dataset.
```julia
using MLJ
using MLJFlux
using Flux
using Plots
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
  model = MultitargetNeuralNetworkRegressor(builder=builder,
                                 rng=123,
                                 epochs=20)
```
For our neural network, since different features likely have different scales, if we do not standardize the network may be implicitly biased towards features with higher magnitudes, or may have [saturated neurons](https://www.informit.com/articles/article.aspx%3fp=3131594&seqNum=2)  and not train well. Therefore, standardization is key!
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
plt = plot()

rates = 10. .^ (-5:0)

foreach(rates) do η
  pipe.transformed_target_model_deterministic.model.optimiser.eta = η
  fit!(mach, force=true, verbosity=0)
  losses =
      report(mach).transformed_target_model_deterministic.model.training_losses[3:end]
  plot!(1:length(losses), losses, label=η)
end
plt #!md

savefig(joinpath("assets", "learning_rate.png"))


pipe.transformed_target_model_deterministic.model.optimiser.eta = 0.0001

```

## Using Iteration Controls

We can also wrap the model with MLJ Iteration controls. Suppose we want a model that trains until the out of sample loss does not improve for 6 epochs. We can use the `NumberSinceBest(6)` stopping criterion. We can also add some extra stopping criterion, `InvalidValue` and `Timelimit(1/60)`, as well as some controls to print traces of the losses. First we can define some methods to initialize or clear the traces as well as updte the traces.
```julia
# For initializing or clearing the traces:

clear() = begin
  global losses = []
  global training_losses = []
  global epochs = []
  return nothing
end

# And to update the traces:

update_loss(loss) = push!(losses, loss)
update_training_loss(report) =
  push!(training_losses,
        report.transformed_target_model_deterministic.model.training_losses[end])
update_epochs(epoch) = push!(epochs, epoch)
```
For further reference of controls, see [the documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/%23Controls-provided). To apply the controls, we simply stack them in a vector and then make an `IteratedModel`:
```julia
controls=[Step(1),
        NumberSinceBest(6),
        InvalidValue(),
        TimeLimit(1/60),
        WithLossDo(update_loss),
        WithReportDo(update_training_loss),
WithIterationsDo(update_epochs)]

iterated_pipe =
  IteratedModel(model=pipe,
                controls=controls,
                resampling=Holdout(fraction_train=0.8),
                measure = l2)
```
Next, we can clear the traces, fit the model, and plot the traces:
```julia
clear()
mach = machine(iterated_pipe, X, y)
fit!(mach)

plot(epochs, losses,
   xlab = "epoch",
   ylab = "mean sum of squares error",
   label="out-of-sample",
   legend = :topleft);
scatter!(twinx(), epochs, training_losses, label="training", color=:red) #!md

savefig(joinpath("assets", "loss.png"))
```

### Brief note on iterated models

Training an `IteratedModel` means holding out some data (80% in this case) so an out-of-sample loss can be tracked and used in the specified stopping criterion, `NumberSinceBest(4)`. However, once the stop is triggered, the model wrapped by `IteratedModel` (our pipeline model) is retrained on all data for the same number of iterations. Calling `predict(mach, Xnew)` on new data uses the updated learned parameters.

## Evaluating Iterated Models

We can evaluate our model with the `evaluate!` function:
```julia
e = evaluate!(mach,
             resampling=CV(nfolds=8),
             measures=[l1, l2])

using Measurements
l1_loss = e.measurement[1] ± std(e.per_fold[1])/sqrt(7)
@show l1_loss
```
We take this estimate of the uncertainty of the generalization error with a [grain of salt](https://direct.mit.edu/neco/article-abstract/10/7/1895/6224/Approximate-Statistical-Tests-for-Comparing)).

## Comparison with other models on the test set

Although we cannot assign them statistical significance, here are comparisons, on the untouched test set, of the eror of our self-iterating neural network regressor with a couple of other models trained on the same data (using default hyperparameters):
```julia

function performance(model)
   mach = machine(model, X, y) |> fit!
   yhat = predict(mach, Xtest)
   l1(yhat, ytest) |> mean
end
performance(iterated_pipe)

three_models = [(@load EvoTreeRegressor)(), # tree boosting model
               (@load LinearRegressor pkg=MLJLinearModels)(),
               iterated_pipe]

errs = performance.(three_models)

(models=MLJ.name.(three_models), mean_square_errors=errs) |> pretty


```
See also
[`NeuralNetworkRegressor`](@ref)
"""
MultitargetNeuralNetworkRegressor

const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}
