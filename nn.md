# NeuralNetworkRegressor

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

- `builder=MLJFlux.Linear(σ=Flux.relu)`: An MLJFlux builder that constructs a neural network. Possible `builders` include: `Linear`, `Short`, and `MLP`. You can construct your own builder using the `@builder` macro, see examples for further information.
- `optimiser::Flux.ADAM()`: A `Flux.Optimise` optimiser. The optimiser performs the updating of the weights of the network. For further reference, see either the examples or [the Flux optimiser documentation](https://fluxml.ai/Flux.jl/stable/training/optimisers/). To choose a learning rate (the update rate of the optimizer), a good rule of thumb is to start out at `10e-3`, and tune using powers of 10 between `1` and `1e-7`.
- `loss=Flux.mse`: The loss function which the network will optimize. Should be a function which can be called in the form `loss(yhat, y)`. Possible loss functions are listed in [the Flux loss function documentation](https://fluxml.ai/Flux.jl/stable/models/losses/). For a regression task, the most natural loss functions are: 
    - `Flux.mse`
    - `Flux.mae`
    - `Flux.msle`
    - `Flux.huber_loss`
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
  deterministic.


# Fitted parameters

The fields of `fitted_params(mach)` are:
- `chain`: The trained "chain", or series of layers, functions, and activations which make up the neural network.


# Report

The fields of `report(mach)` are:
- `training_losses`: The history of training losses, a vector containing the history of all the losses during training. The first element of the vector is the initial penalized loss. After the first element, the nth element corresponds to the loss of epoch n-1.

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
Since MLJFlux models do not handle ordered factos, we can treat `:RAD` as `Continuous`:
```julia
X = coerce(X, :RAD=>Continuous)
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

  NeuralNetworkRegressor = @load NeuralNetworkRegressor
      model = NeuralNetworkRegressor(builder=builder,
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

#-

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
