# # Using MLJ to classifiy the MNIST image dataset

using Pkg
const DIR = @__DIR__
Pkg.activate(DIR)
Pkg.instantiate()

# **Julia version** is assumed to be 1.6.*

using MLJ
using Flux
import MLJFlux
using Random
Random.seed!(123)

MLJ.color_off()

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));

# ## Basic training

# Downloading the MNIST image dataset:

import MLDatasets

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
images, labels = MNIST.traindata();

# In MLJ, integers cannot be used for encoding categorical data, so we
# must force the labels to have the `Multiclass` [scientific
# type](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/). For
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
# data](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/#Type-coercion-for-image-data-1)

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

flatten(x::AbstractArray) = reshape(x, :, size(x)[end])
half(x) = div(x, 2)

function MLJFlux.build(b::MyConvBuilder, n_in, n_out, n_channels)

    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3

    mod(k, 2) == 1 || error("`filter_size` must be odd. ")

    p = div(k - 1, 2) # padding to preserve image size on convolution:

    h = n_in[1] |> half |> half |> half # final "image" height
    w = n_in[2] |> half |> half |> half # final "image" width

    return Chain(
        Conv((k, k), n_channels => c1, pad=(p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad=(p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad=(p, p), relu),
        MaxPool((2 ,2)),
        flatten,
        Dense(h*w*c3, n_out))
end

# **Note.** There is no final `softmax` here, as this is applied by
# default in all MLJFLux classifiers. Customisation of this behaviour
# is controlled using using the `finaliser` hyperparameter of the
# classifier.

# We now define the MLJ model. If you have a GPU, substitute
# `acceleration=CUDALibs()` below:

ImageClassifier = @load ImageClassifier
clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
                      acceleration=CPU1(),
                      batch_size=50,
                      epochs=10)

# You can add Flux options `optimiser=...` and `loss=...` here. At
# present, `loss` must be a Flux-compatible loss, not an MLJ measure.

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
cross_entropy(predicted_labels, labels[501:1000]) |> mean

# Or, in one line (after resetting the RNG seed to ensure the same
# result):

Random.seed!(123)
evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)


# ## Wrapping the MLJFlux model with iteration controls

# Any iterative MLJ model implementing the warm restart functionality
# illustrated above for `ImageClassifier` can be wrapped in *iteration
# controls*, as we demonstrate next. For more on MLJ's
# `IteratedModel` wrapper, see the [MLJ
# documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/).

# The "self-iterating" model, called `imodel` below, is for iterating the
# image classifier defined above until one of the following stopping
# criterion apply:

# - `Patience(3)` (3 consecutive increases in the loss)

# - `InvalidValue()` (an out-of-sample loss, or a training loss,
#   is `NaN`, `Inf`, or `-Inf`)

# - `TimeLimit(t=1/60)` (training time has exceeded one minute)

# Additionally, training a machine bound to `imodel` will:

# - save a snapshot of the machine every three epochs

# - record the out-of-sample loss and training losses for plotting

# For a complete list of controls, see [this
# table](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided).

losses = []
training_losses = [];

add_loss(loss) = push!(losses, loss)
add_training_loss(losses) = push!(training_losses, losses[end])

imodel = IteratedModel(model=clf,
                       controls=[Step(1),      # train one epoch at a time
                                 Patience(2),
                                 InvalidValue(),
                                 TimeLimit(0.5),
                                 Save(joinpath(DIR, "mnist_machine.jlso")),
                                 WithLossDo(), # for logging to `Info`
                                 WithLossDo(add_loss),
                                 WithTrainingLossesDo(add_training_loss)],
                       resampling=Holdout(fraction_train=0.7),
                       measure=log_loss,
                       retrain=false)


# Binding our self-iterating model to data:

mach = machine(imodel, images, labels)

# And training on the first 500 images:

fit!(mach, rows=1:500)

# A comparison of the training and out-of-sample losses:

plot(losses,
     title="Cross Entropy",
     xlab = "epoch",
     label="out-of-sample")
plot!(training_losses, label="training")

# Retrieving a snapshot for a prediction:

mach2 = machine(joinpath(DIR, "mnist_machine5.jlso"))
predict_mode(mach2, images[501:503])

#-

using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=false) #src
Literate.notebook(@__FILE__, @__DIR__, execute=false) #src

