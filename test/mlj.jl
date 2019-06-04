# this file has MLJ dependency. Not intended to be part of final
# testing?

## REGRESSOR DEMO

# using Revise
using MLJ
import Flux
using FluxMLJ
using CategoricalArrays

task = load_boston()
X_, y_ = task()

# whiten target:
ustand_model = UnivariateStandardizer()
ustand = machine(ustand_model, y_)
fit!(ustand)
y = transform(ustand, y_);

# whiten inputs:
stand_model = Standardizer()
stand = machine(stand_model, X_)
fit!(stand)
X = transform(stand, X_);

# construct a neural network model and inspect all hyperparameters:
builder = FluxMLJ.Short()
optimiser = Flux.Momentum(0.003)
nnmodel = NeuralNetworkRegressor(builder=builder, optimiser=optimiser)
params(nnmodel)


# learning curves:
nn = machine(nnmodel, X, y)
r = range(nnmodel, :n, lower=1, upper=50)
curve = learning_curve!(nn, nested_range=(n=r,), n=4)

using Plots
plotly()

plot(curve.parameter_values, curve.measurements,
     xlab=curve.parameter_name)


## TUNING DEMO (bit contrived)

nnmodel.optimiser_changes_trigger_retraining=true

# define some hyperparameters ranges:
r1 = range(builder, :n_hidden, lower=1, upper=5)
r2 = range(optimiser, :eta, lower = 0.001, upper = 0.01, scale=:log)

# assemble ranges into a named tuple matching pattern of
# params(nnmodel):
nested_ranges = (builder=(n_hidden=r1,), optimiser=(eta=r2,))

# wrap in tuning strategy:
tuned_nnmodel = TunedModel(model=nnmodel,
                           tuning=Grid(resolution=5),
                           measure=rms,
                           nested_ranges=nested_ranges)

# tune the neural network:
tuned = machine(tuned_nnmodel, X, y)
fit!(tuned, verbosity=2)
report(tuned)
plot(tuned)
# uncomment 3 lines to visualize tuning:
# plot(tuned)


## CLASSIFIER DEMO

task = load_iris()
X_, y = task()

# whiten inputs:
stand_model = Standardizer()
stand = machine(stand_model, X_)
fit!(stand)
X = transform(stand, X_);

# construct a neural network model and inspect all hyperparameters:
builder = FluxMLJ.Short()
optimiser = Flux.Momentum(0.003)
nnmodel = NeuralNetworkClassifier(builder=builder, optimiser=optimiser)
params(nnmodel)


# learning curves:
nn = machine(nnmodel, X, y)
r = range(nnmodel, :n, lower=1, upper=50)
curve = learning_curve!(nn, nested_range=(n=r,), n=4)

using Plots
plotly()

plot(curve.parameter_values, curve.measurements,
     xlab=curve.parameter_name)



## CRAP

N = 200
X = MLJ.table(randn(10N, 5))

# while multivariate targets are vectors of tuples:
yvector = 1 .+ X.x1 - X.x2 .+ 1 .- 2X.x4 + X.x5
train = 1:7N
test = (7N+1):10N
yvector = yvector ./ maximum(yvector)

l = ["1", "2", "3", "4"]

function get_labels(ele)
    if ele >0.5
        return "1"
    elseif ele > 0.0
        return "2"
    elseif ele > -0.5
        return "3"
    else
        return "4"
    end
end

y = CategoricalArray(get_labels.(yvector));

builder = FluxMLJ.Linear(σ=Flux.sigmoid)
nnmodel = FluxMLJ.NeuralNetworkClassifier(loss=Flux.crossentropy,
                                        builder=builder)
params(nnmodel)

nn = machine(nnmodel, X, y)
fit!(nn)

r = range(nnmodel, :n, lower=1, upper=50)
curve = learning_curve!(nn, nested_range=(n=r,), n=4, measure=cross_entropy)

using Plots
plotly()

plot(curve.parameter_values, curve.measurements,
     xlab=curve.parameter_name)
