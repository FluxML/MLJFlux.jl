# # Using MLJ to classifiy the MNIST image dataset

using Pkg
const DIR = @__DIR__
Pkg.activate(DIR)
Pkg.instantiate()

# **Julia version** is assumed to be ^1.10

using MLJ
using Flux
import MLJFlux
import MLUtils
import MLJIteration # for `skip`

# If running on a GPU, you will also need to `import CUDA` and `import cuDNN`.

using Plots
gr(size=(600, 300*(sqrt(5)-1)));

# ## Basic training

# Downloading the MNIST image dataset:

import MLDatasets: MNIST

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
images, labels = MNIST(split=:train)[:];

# In MLJ, integers cannot be used for encoding categorical data, so we
# must force the labels to have the `Multiclass` [scientific
# type](https://juliaai.github.io/ScientificTypes.jl/dev/). For
# more on this, see [Working with Categorical
# Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/).

labels = coerce(labels, Multiclass);
images = coerce(images, GrayImage);

# Checking scientific types:

@assert scitype(images) <: AbstractVector{<:Image}
@assert scitype(labels) <: AbstractVector{<:Finite}

# Looks good.

# For general instructions on coercing image data, see [Type coercion
# for image
# data](https://juliaai.github.io/ScientificTypes.jl/dev/#Type-coercion-for-image-data)

images[1]

# We start by defining a suitable `Builder` object. This is a recipe
# for building the neural network. Our builder will work for images of
# any (constant) size, whether they be color or black and white (ie,
# single or multi-channel).  The architecture always consists of six
# alternating convolution and max-pool layers, and a final dense
# layer; the filter size and the number of channels after each
# convolution layer is customisable.

import MLJFlux
struct MyConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end

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
        MLUtils.flatten)
    d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
    return Chain(front, Dense(d, n_out, init=init))
end

# **Note.** There is no final `softmax` here, as this is applied by
# default in all MLJFLux classifiers. Customisation of this behaviour
# is controlled using using the `finaliser` hyperparameter of the
# classifier.

# We now define the MLJ model.

ImageClassifier = @load ImageClassifier
clf = ImageClassifier(
    builder=MyConvBuilder(3, 16, 32, 32),
    batch_size=50,
    epochs=10,
    rng=123,
)

# You can add Flux options `optimiser=...` and `loss=...` here. At
# present, `loss` must be a Flux-compatible loss, not an MLJ
# measure. To run on a GPU, set `acceleration=CUDALib()` and omit `rng`.

# Binding the model with data in an MLJ machine:
mach = machine(clf, images, labels);

# Training for 10 epochs on the first 500 images:

fit!(mach, rows=1:500, verbosity=2);

# Inspecting:

report(mach)

#-

chain = fitted_params(mach)

#-

Flux.params(chain)[2]

#-

# Adding 20 more epochs:

clf.epochs = clf.epochs + 20
fit!(mach, rows=1:500);

# Computing an out-of-sample estimate of the loss:

predicted_labels = predict(mach, rows=501:1000);
cross_entropy(predicted_labels, labels[501:1000])

# Or to fit and predict, in one line:

evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)


# ## Wrapping the MLJFlux model with iteration controls

# Any iterative MLJFlux model can be wrapped in *iteration controls*,
# as we demonstrate next. For more on MLJ's `IteratedModel` wrapper,
# see the [MLJ
# documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/).

# The "self-iterating" classifier, called `iterated_clf` below, is for
# iterating the image classifier defined above until one of the
# following stopping criterion apply:

# - `Patience(3)`: 3 consecutive increases in the loss
# - `InvalidValue()`: an out-of-sample loss, or a training loss, is `NaN`, `Inf`, or `-Inf`
# - `TimeLimit(t=5/60)`: training time has exceeded 5 minutes
#
# These checks (and other controls) will be applied every two epochs
# (because of the `Step(2)` control). Additionally, training a
# machine bound to `iterated_clf` will:
#
# - save a snapshot of the machine every three control cycles (every six epochs)
# - record traces of the out-of-sample loss and training losses for plotting
# - record mean value traces of each Flux parameter for plotting

# For a complete list of controls, see [this
# table](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided).

# ### Wrapping the classifier

# Some helpers

make2d(x::AbstractArray) = reshape(x, :, size(x)[end])
make1d(x::AbstractArray) = reshape(x, length(x));

# To extract Flux params from an MLJFlux machine

parameters(mach) = make1d.(Flux.params(fitted_params(mach)));

# To store the traces:

losses = []
training_losses = []
parameter_means = Float32[];
epochs = []

# To update the traces:

update_loss(loss) = push!(losses, loss)
update_training_loss(losses) = push!(training_losses, losses[end])
update_means(mach) = append!(parameter_means, mean.(parameters(mach)));
update_epochs(epoch) = push!(epochs, epoch)

# The controls to apply:

save_control =
    MLJIteration.skip(Save(joinpath(DIR, "mnist.jls")), predicate=3)

controls=[
    Step(2),
    Patience(3),
    InvalidValue(),
    TimeLimit(5/60),
    save_control,
    WithLossDo(),
    WithLossDo(update_loss),
    WithTrainingLossesDo(update_training_loss),
    Callback(update_means),
    WithIterationsDo(update_epochs),
];

# The "self-iterating" classifier:

iterated_clf = IteratedModel(
    clf,
    controls=controls,
    resampling=Holdout(fraction_train=0.7),
    measure=log_loss,
)

# ### Binding the wrapped model to data:

mach = machine(iterated_clf, images, labels);


# ### Training

fit!(mach, rows=1:500);

# ### Comparison of the training and out-of-sample losses:

plot(
    epochs,
    losses,
    xlab = "epoch",
    ylab = "cross entropy",
    label="out-of-sample",
)
plot!(epochs, training_losses, label="training")

savefig(joinpath(DIR, "loss.png"))

# ### Evolution of weights

n_epochs =  length(losses)
n_parameters = div(length(parameter_means), n_epochs)
parameter_means2 = reshape(copy(parameter_means), n_parameters, n_epochs)'
plot(
    epochs,
    parameter_means2,
    title="Flux parameter mean weights",
    xlab = "epoch",
)

# **Note.** The higher the number, the deeper the chain parameter.

savefig(joinpath(DIR, "weights.png"))


# ### Retrieving a snapshot for a prediction:

mach2 = machine(joinpath(DIR, "mnist3.jls"))
predict_mode(mach2, images[501:503])


# ### Restarting training

# Mutating `iterated_clf.controls` or `clf.epochs` (which is otherwise
# ignored) will allow you to restart training from where it left off.

iterated_clf.controls[2] = Patience(4)
fit!(mach, rows=1:500)

plot(
    epochs,
    losses,
    xlab = "epoch",
    ylab = "cross entropy",
    label="out-of-sample",
)
plot!(epochs, training_losses, label="training")
