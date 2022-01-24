# # Building an MLJFlux regression model for the Boston house
# # price dataset

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# **Julia version** is assumed to be 1.6.*

using MLJ
using MLJFlux
using Flux
using Plots

# This tutorial uses MLJ's `IteratedModel` wrapper to transform the
# MLJFlux `NeuralNetworkRegressor` into a model that **automatically
# selects the number of epochs** required to optimize an out-of-sample
# loss.

# We also show how to include the model in a **pipeline** to carry out
# standardization of the features and target.


# ## Loading data

data = OpenML.load(531); # Loads from https://www.openml.org/d/531

# The target `y` is `:MEDV` and everything else except `:CHAS` goes
# into the features `X`:

y, X = unpack(data, ==(:MEDV), !=(:CHAS); rng=123);

# We specified the seed `rng` to shuffle the observations. The Charles
# River dummy variable `:CHAS` is dropped, as not deemed to be
# relevant.

# Inspecting the scientific types:

scitype(y)

#-

schema(X)

# We'll regard `:RAD` (index of accessibility to radial highways) as
# `Continuous` as MLJFlux models don't handle ordered factors:

X = coerce(X, :RAD => Continuous);

# Let's split off a test set for final testing:

(X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);


# ## Defining a builder

# In the macro call below, `n_in` is expected to represent the number
# of inputs features and `rng` a RNG (builders are generic, ie can be
# applied to data with any number of input features):

builder = MLJFlux.@builder begin
    init=Flux.glorot_uniform(rng)
    Chain(Dense(n_in, 64, relu, init=init),
          Dense(64, 32, relu, init=init),
          Dense(32, 1, init=init))
end


# ## Defining a MLJFlux model:

NeuralNetworkRegressor = @load NeuralNetworkRegressor
    model = NeuralNetworkRegressor(builder=builder,
                                   rng=123,
                                   epochs=20)


# ## Standardization

# The following wraps our regressor in feature and target standardizations:

pipe = Standardizer |> TransformedTargetModel(model, target=Standardizer)

# Notice that our original neural network model is now a
# hyper-parameter of the composite `pipe`, with the automatically
# generated name, `:neural_network_regressor`.


# ## Choosing a learning rate

# Let's see how the training losses look for the default optimiser. For
# MLJFlux models, `fit!` will print these losses if we bump the
# verbosity level (default is always 1):

mach = machine(pipe, X, y)
fit!(mach, verbosity=2)

# They are also extractable from the training report (which includes
# the pre-train loss):

report(mach).transformed_target_model_deterministic.model.training_losses

# Next, let's visually compare a few learning rates:

plt = plot()
rates = [5e-5, 1e-4, 0.005, 0.001, 0.05]

# By default, changing only the optimiser will not trigger a
# cold-restart when we `fit!` (to allow for adaptive learning rate
# control). So we call `fit!` with the `force=true`
# option. (Alternatively, one can change the hyper-parameter
# `pipe.neural_network_regressor.optimiser_changes_trigger_retraining`
# to `true`.)

# We'll skip the first few losses to get a better vertical scale in
# our plot.

foreach(rates) do η
    pipe.transformed_target_model_deterministic.model.optimiser.eta = η
    fit!(mach, force=true, verbosity=0)
    losses =
        report(mach).transformed_target_model_deterministic.model.training_losses[3:end]
    plot!(1:length(losses), losses, label=η)
end
plt #!md

#-

savefig(joinpath("assets", "learning_rate.png"))

# ![](assets/learing_rate.png) #md

# We'll go with the second most conservative rate for now:

pipe.transformed_target_model_deterministic.model.optimiser.eta = 0.0001


# ## Wrapping in iteration control

# We want a model that trains until an out-of-sample loss satisfies
# the `NumberSinceBest(6)` stopping criterion. We'll add some fallback
# stopping criterion `InvalidValue` and `TimeLimit(1/60)`, and
# controls to print traces of the losses.

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

# The controls to apply (see
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided)
# for the complete list):

controls=[Step(1),
          NumberSinceBest(6),
          InvalidValue(),
          TimeLimit(1/60),
          WithLossDo(update_loss),
          WithReportDo(update_training_loss),
          WithIterationsDo(update_epochs)]

# Next we create a "self-iterating" version of the pipeline. Note
# that the iteration parameter is a nested hyperparameter:

iterated_pipe =
    IteratedModel(model=pipe,
                  controls=controls,
                  resampling=Holdout(fraction_train=0.8),
                  iteration_parameter=:(neural_network_regressor.epochs),
                  measure = l2)

# Training the wrapped model on all the train/validation data:

clear()
mach = machine(iterated_pipe, X, y)
fit!(mach)

# And plotting the traces:

plot(epochs, losses,
     xlab = "epoch",
     ylab = "mean sum of squares error",
     label="out-of-sample",
     legend = :topleft);
scatter!(twinx(), epochs, training_losses, label="training", color=:red) #!md

#-

savefig(joinpath("assets", "loss.png"))

# ![](assets/losses.png) #md

# **How `IteratedModel` works.** Training an `IteratedModel` means
# holding out some data (80% in this case) so an out-of-sample loss
# can be tracked and used in the specified stopping criterion,
# `NumberSinceBest(4)`. However, once the stop is triggered, the model
# wrapped by `IteratedModel` (our pipeline model) is retrained on all
# data for the same number of iterations. Calling `predict(mach,
# Xnew)` on new data uses the updated learned parameters.

# In other words, `iterated_model` is a "self-iterating" version of
# the original model, where `epochs` has been transformed from
# hyper-parameter to *learned* parameter.


# ## An evaluation of the self-iterating model

# Here's an estimate of performance of our "self-iterating"
# model:

e = evaluate!(mach,
              resampling=CV(nfolds=8),
              measures=[l1, l2])

#-

using Measurements
l1_loss = e.measurement[1] ± std(e.per_fold[1])/sqrt(7)
@show l1_loss

# We take this estimate of the uncertainty of the generalization error with a [grain of salt](https://direct.mit.edu/neco/article-abstract/10/7/1895/6224/Approximate-Statistical-Tests-for-Comparing)).


# ## Comparison with other models on the test set

# Although we cannot assign them statistical significance, here are
# comparisons, on the untouched test set, of the eror of our
# self-iterating neural network regressor with a couple of other
# models trained on the same data (using default hyperparameters):

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
