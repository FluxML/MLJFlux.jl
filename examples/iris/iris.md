```@meta
EditURL = "<unknown>/../../MLJFlux/examples/iris/iris.jl"
```

# Using MLJ with Flux to train the iris dataset

```@example iris
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
```

**Julia version** is assumed to be 1.6.*

```@example iris
using MLJ
using Flux
import RDatasets

using Random
Random.seed!(123)

MLJ.color_off()

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));
nothing #hide
```

Following is a very basic introductory example, using a default
builder and no standardization of input features.

For a more advanced illustration, see the [MNIST images
example](https://github.com/FluxML/MLJFlux.jl/blob/dev/examples/mnist).

## Loading some data and instantiating a model

```@example iris
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
NeuralNetworkClassifier = @load NeuralNetworkClassifier
clf = NeuralNetworkClassifier()
```

## Incremental training

```@example iris
import Random.seed!; seed!(123)
mach = machine(clf, X, y)
fit!(mach)

training_loss = cross_entropy(predict(mach, X), y) |> mean
```

Increasing learning rate and adding iterations:

```@example iris
clf.optimiser.eta = clf.optimiser.eta * 2
clf.epochs = clf.epochs + 5

fit!(mach, verbosity=2);
nothing #hide
```

```@example iris
training_loss = cross_entropy(predict(mach, X), y) |> mean
```

## Accessing the Flux chain (model)

```@example iris
chain = fitted_params(mach).chain
```

##  Evolution of out-of-sample performance

```@example iris
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

```@example iris
savefig("iris_history.png")
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

