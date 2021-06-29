```@meta
EditURL = "<unknown>/../../MLJFlux/examples/mnist/mnist.jl"
```

# Using MLJ to classifiy the MNIST image dataset

```@example mnist
using Pkg
const DIR = @__DIR__
Pkg.activate(DIR)
Pkg.instantiate()
```

**Julia version** is assumed to be ^1.6

```@example mnist
using MLJ
using Flux
import MLJFlux
import MLJIteration # for `skip`

MLJ.color_off()

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));
nothing #hide
```

## Basic training

Downloading the MNIST image dataset:

```@example mnist
import MLDatasets: MNIST

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
images, labels = MNIST.traindata();
nothing #hide
```

In MLJ, integers cannot be used for encoding categorical data, so we
must force the labels to have the `Multiclass` [scientific
type](https://juliaai.github.io/ScientificTypes.jl/dev/). For
more on this, see [Working with Categorical
Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/).

```@example mnist
labels = coerce(labels, Multiclass);
images = coerce(images, GrayImage);
nothing #hide
```

Checking scientific types:

```@example mnist
@assert scitype(images) <: AbstractVector{<:Image}
@assert scitype(labels) <: AbstractVector{<:Finite}
```

Looks good.

For general instructions on coercing image data, see [Type coercion
for image
data](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/#Type-coercion-for-image-data-1)

```@example mnist
images[1]
```

We start by defining a suitable `Builder` object. This is a recipe
for building the neural network. Our builder will work for images of
any (constant) size, whether they be color or black and white (ie,
single or multi-channel).  The architecture always consists of six
alternating convolution and max-pool layers, and a final dense
layer; the filter size and the number of channels after each
convolution layer is customisable.

```@example mnist
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

**Note.** There is no final `softmax` here, as this is applied by
default in all MLJFLux classifiers. Customisation of this behaviour
is controlled using using the `finaliser` hyperparameter of the
classifier.

We now define the MLJ model. If you have a GPU, substitute
`acceleration=CUDALibs()` below:

```@example mnist
ImageClassifier = @load ImageClassifier
clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
                      batch_size=50,
                      epochs=10,
                      rng=123)
```

You can add Flux options `optimiser=...` and `loss=...` here. At
present, `loss` must be a Flux-compatible loss, not an MLJ
measure. To run on a GPU, set `acceleration=CUDALib()`.

Binding the model with data in an MLJ machine:

```@example mnist
mach = machine(clf, images, labels);
nothing #hide
```

Training for 10 epochs on the first 500 images:

```@example mnist
fit!(mach, rows=1:500, verbosity=2);
nothing #hide
```

Inspecting:

```@example mnist
report(mach)
```

```@example mnist
chain = fitted_params(mach)
```

```@example mnist
Flux.params(chain)[2]
```

Adding 20 more epochs:

```@example mnist
clf.epochs = clf.epochs + 20
fit!(mach, rows=1:500);
nothing #hide
```

Computing an out-of-sample estimate of the loss:

```@example mnist
predicted_labels = predict(mach, rows=501:1000);
cross_entropy(predicted_labels, labels[501:1000]) |> mean
```

Or, in one line:

```@example mnist
evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)
```

## Wrapping the MLJFlux model with iteration controls

Any iterative MLJFlux model can be wrapped in *iteration controls*,
as we demonstrate next. For more on MLJ's `IteratedModel` wrapper,
see the [MLJ
documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/).

The "self-iterating" classifier, called `iterated_clf` below, is for
iterating the image classifier defined above until one of the
following stopping criterion apply:

- `Patience(3)`: 3 consecutive increases in the loss
- `InvalidValue()`: an out-of-sample loss, or a training loss, is `NaN`, `Inf`, or `-Inf`
- `TimeLimit(t=5/60)`: training time has exceeded 5 minutes

These checks (and other controls) will be applied every two epochs
(because of the `Step(2)` control). Additionally, training a
machine bound to `iterated_clf` will:

- save a snapshot of the machine every three control cycles (every six epochs)
- record traces of the out-of-sample loss and training losses for plotting
- record mean value traces of each Flux parameter for plotting

For a complete list of controls, see [this
table](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided).

### Wrapping the classifier

Some helpers

```@example mnist
make2d(x::AbstractArray) = reshape(x, :, size(x)[end])
make1d(x::AbstractArray) = reshape(x, length(x));
nothing #hide
```

To extract Flux params from an MLJFlux machine

```@example mnist
parameters(mach) = make1d.(Flux.params(fitted_params(mach)));
nothing #hide
```

To store the traces:

```@example mnist
losses = []
training_losses = []
parameter_means = Float32[];
nothing #hide
```

To update the traces:

```@example mnist
update_loss(loss) = push!(losses, loss)
update_training_loss(losses) = push!(training_losses, losses[end])
update_means(mach) = append!(parameter_means, mean.(parameters(mach)));
nothing #hide
```

The controls to apply:

```@example mnist
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
          Callback(update_means)
];
nothing #hide
```

The "self-iterating" classifier:

```@example mnist
iterated_clf = IteratedModel(model=clf,
                       controls=controls,
                       resampling=Holdout(fraction_train=0.7),
                       measure=log_loss)
```

### Binding the wrapped model to data:

```@example mnist
mach = machine(iterated_clf, images, labels);
nothing #hide
```

### Training

```@example mnist
fit!(mach, rows=1:500);
nothing #hide
```

### Comparison of the training and out-of-sample losses:

```@example mnist
plot(losses,
     xlab = "epoch",
     ylab = "root squared error",
     label="out-of-sample")
plot!(training_losses, label="training")

savefig(joinpath(DIR, "loss.png"))
```

### Evolution of weights

```@example mnist
n_epochs =  length(losses)
n_parameters = div(length(parameter_means), n_epochs)
parameter_means2 = reshape(copy(parameter_means), n_parameters, n_epochs)'
plot(parameter_means2,
     title="Flux parameter mean weights",
     xlab = "epoch")
```

**Note.** The the higher the number, the deeper the chain parameter.

```@example mnist
savefig(joinpath(DIR, "weights.png"))
```

### Retrieving a snapshot for a prediction:

```@example mnist
mach2 = machine(joinpath(DIR, "mnist3.jlso"))
predict_mode(mach2, images[501:503])
```

### Restarting training

Mutating `iterated_clf.controls` or `clf.epochs` (which is otherwise
ignored) will allow you to restart training from where it left off.

```@example mnist
iterated_clf.controls[2] = Patience(4)
fit!(mach, rows=1:500)

plot(losses,
     xlab = "epoch",
     ylab = "root squared error",
     label="out-of-sample")
plot!(training_losses, label="training")
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

