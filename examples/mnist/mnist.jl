# # Using MLJ to classifiy the MNIST image dataset

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

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

import Flux.Data.MNIST
images, labels = MNIST.images(), MNIST.labels();

# In MLJ, integers cannot be used for encoding categorical data, so we
# must force the labels to have the `Multiclass` [scientific
# type](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/). For
# more on this, see [Working with Categorical
# Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/).

labels = coerce(labels, Multiclass);

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
mutable struct MyConvBuilder
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


# ## Using out-of-sample loss estimates to terminate training:

# MLJ will eventually provide model wrappers for controlling iterative
# models. In the meantime some control can be implememted using the
# [EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl)
# package, and without the usual need for callbacks.

# Defining an `EarlyStopper` object combining three separate stopping
# critera:

using EarlyStopping
stopper = EarlyStopper(NotANumber(), Patience(3), UP())

losses = Float32[]
training_losses = Float32[];

# Resetting the number of epochs to zero:

clf.epochs = 0;

# Defining a function to increment the number of epochs, re-evaluate,
# and test for early stopping:

function done()
    clf.epochs = clf.epochs + 1
    e = evaluate!(mach,
                  resampling=Holdout(fraction_train=0.5),
                  measure=cross_entropy,
                  rows=1:1000,
                  verbosity=0)
    loss = e.measurement[1][1]
    push!(losses, loss)
    training_loss = report(mach).training_losses[end]
    push!(training_losses, training_loss)
    println("out-of-sample loss: $loss")
    return done!(stopper, loss)
end;

# **Note.** Each time the number of epochs is increased and
# `evaluate!` is called, warm-start training is used (assuming
# `resampling isa Holdout`). This is because MLJ machines cache
# hyper-parameters and learned parameters to avoid unnecessary
# retraining. In other frameworks the same behaviour is implemented
# using callbacks, but we don't need this here.

while !done() end
message(stopper)

# A comparison of the training and out-of-sample losses:
plot(losses,
     title="Cross Entropy",
     xlab = "epoch",
     label="out-of-sample")
plot!(training_losses, label="training")

using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=true) #src
Literate.notebook(@__FILE__, @__DIR__, execute=true) #src
