# this file has MLJ dependency. Not intended to be part of final
# testing?

# using Revise
using MLJ
import Flux
using FluxMLJ

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
optimiser = Flux.Momentum()
nnmodel = NeuralNetworkRegressor(builder=builder, optimiser=optimiser)
params(nnmodel)

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
fit!(tuned)
report(tuned)

# uncomment 3 lines to visualize tuning:
# using Plots
# plotly()
# plot(tuned)
