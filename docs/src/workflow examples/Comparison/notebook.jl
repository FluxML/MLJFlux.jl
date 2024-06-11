# # Model Comparison with MLJFlux

# In this workflow example, we see how we can compare different machine learning models
# with a neural network from MLJFlux.
using Pkg     #!md
Pkg.activate(@__DIR__);     #!md
Pkg.instantiate();     #!md

# **Julia version** is assumed to be 1.10.*

# ### Basic Imports

using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import RDatasets        # Dataset source
using DataFrames        # To visualize hyperparameter search results
import Optimisers       # native Flux.jl optimisers no longer supported

# ### Loading and Splitting the Data

iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);


# ### Instantiating the models Now let's construct our model. This follows a similar setup
# to the one followed in the [Quick Start](../../index.md#Quick-Start).

NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

clf1 = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Optimisers.Adam(0.01),
    batch_size=8,
    epochs=50,
    rng=42
    )

# Let's as well load and construct three other classical machine learning models:
BayesianLDA = @load BayesianLDA pkg=MultivariateStats
clf2 = BayesianLDA()
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
clf3 = RandomForestClassifier()
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost
clf4 = XGBoostClassifier();


# ### Wrapping One of the Models in a TunedModel Instead of just comparing with four
# models with the default/given hyperparameters, we will give `XGBoostClassifier` an
# unfair advantage By wrapping it in a `TunedModel` that considers the best learning rate
# η for the model.

r1 = range(clf4, :eta, lower=0.01, upper=0.5, scale=:log10)
tuned_model_xg = TunedModel(
    model=clf4,
    ranges=[r1],
    tuning=Grid(resolution=10),
    resampling=CV(nfolds=5, rng=42),
    measure=cross_entropy,
);

# Of course, one can wrap each of the four in a TunedModel if they are interested in
# comparing the models over a large set of their hyperparameters.

# ### Comparing the models
# We simply pass the four models to the `models` argument of the `TunedModel` construct
tuned_model = TunedModel(
    models=[clf1, clf2, clf3, tuned_model_xg],
    tuning=Explicit(),
    resampling=CV(nfolds=5, rng=42),
    measure=cross_entropy,
);

# Then wrapping our tuned model in a machine and fitting it.
mach = machine(tuned_model, X, y);
fit!(mach, verbosity=0);

# Now let's see the history for more details on the performance for each of the models
history = report(mach).history
history_df = DataFrame(
    mlp = [x[:model] for x in history],
    measurement = [x[:measurement][1] for x in history],
)
sort!(history_df, [order(:measurement)])

# This is Occam's razor in practice.
