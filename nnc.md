# NeuralNetworkClassifier

`NeuralNetworkClassifier`: 
- TODO

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
mach = machine(model, X, y)
Where
- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Finite` with `n_out` classes; check the scitype with `scitype(y)`


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
training_loss = cross_entropy(predict(mach, X), y) |> mean
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
