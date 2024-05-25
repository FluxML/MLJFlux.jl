#### Incremental training

```julia
import Random.seed!; seed!(123)
mach = machine(clf, X, y)
fit!(mach)

julia> training_loss = cross_entropy(predict(mach, X), y) |> mean
0.9064070459118777

# Increasing learning rate and adding iterations:
clf.optimiser.eta = clf.optimiser.eta * 2
clf.epochs = clf.epochs + 5

julia> fit!(mach, verbosity=2)
[ Info: Updating Machine{NeuralNetworkClassifier{Short,…},…} @804.
[ Info: Loss is 0.8686
[ Info: Loss is 0.8228
[ Info: Loss is 0.7706
[ Info: Loss is 0.7565
[ Info: Loss is 0.7347
Machine{NeuralNetworkClassifier{Short,…},…} @804 trained 2 times; caches data
  args:
	1:  Source @985 ⏎ `Table{AbstractVector{Continuous}}`
	2:  Source @367 ⏎ `AbstractVector{Multiclass{3}}`

julia> training_loss = cross_entropy(predict(mach, X), y) |> mean
0.7347092796453824
```

#### Accessing the Flux chain (model)

```julia
julia> fitted_params(mach).chain
Chain(Chain(Dense(4, 3, σ), Flux.Dropout{Float64}(0.5, false), Dense(3, 3)), softmax)
```

####  Evolution of out-of-sample performance

```julia
r = range(clf, :epochs, lower=1, upper=200, scale=:log10)
curve = learning_curve(
    clf, X, y;
    range=r,
    resampling=Holdout(fraction_train=0.7),
    measure=cross_entropy,
    )
using Plots
plot(
    curve.parameter_values,
    curve.measurements,
    xlab=curve.parameter_name,
    xscale=curve.parameter_scale,
    ylab = "Cross Entropy",
    )

```

![](../../../examples/iris/iris_history.png)

This is still work-in-progress. Expect more workflow examples and tutorials soon.