```@meta
EditURL = "comparison.jl"
```

# Model Comparison with MLJFlux

In this workflow example, we see how we can compare different machine learning models with a neural network from MLJFlux.

**Julia version** is assumed to be 1.10.*

### Basic Imports

````julia
using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import RDatasets        # Dataset source
using DataFrames        # To visualize hyperparameter search results
````

### Loading and Splitting the Data

````julia
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
````

### Instantiating the models
Now let's construct our model. This follows a similar setup to the one followed in the [Quick Start](../../index.md).

````julia
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

clf1 = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Flux.ADAM(0.01),
    batch_size=8,
    epochs=50,
    rng=42
    )
````

````
NeuralNetworkClassifier(
  builder = MLP(
        hidden = (5, 4), 
        σ = NNlib.relu), 
  finaliser = NNlib.softmax, 
  optimiser = Adam(0.01, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
  loss = Flux.Losses.crossentropy, 
  epochs = 50, 
  batch_size = 8, 
  lambda = 0.0, 
  alpha = 0.0, 
  rng = 42, 
  optimiser_changes_trigger_retraining = false, 
  acceleration = CPU1{Nothing}(nothing))
````

Let's as well load and construct three other classical machine learning models:

````julia
BayesianLDA = @load BayesianLDA pkg=MultivariateStats
clf2 = BayesianLDA()
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
clf3 = RandomForestClassifier()
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost
clf4 = XGBoostClassifier();
````

````
[ Info: For silent loading, specify `verbosity=0`. 
import MLJMultivariateStatsInterface ✔
[ Info: For silent loading, specify `verbosity=0`. 
import MLJDecisionTreeInterface ✔
[ Info: For silent loading, specify `verbosity=0`. 
import MLJXGBoostInterface ✔

````

### Wrapping One of the Models in a TunedModel
Instead of just comparing with four models with the default/given hyperparameters, we will give `XGBoostClassifier` an unfair advantage
By wrapping it in a `TunedModel` that considers the best learning rate η for the model.

````julia
r1 = range(clf4, :eta, lower=0.01, upper=0.5, scale=:log10)
tuned_model_xg = TunedModel(
    model=clf4,
    ranges=[r1],
    tuning=Grid(resolution=10),
    resampling=CV(nfolds=5, rng=42),
    measure=cross_entropy,
);
````

Of course, one can wrap each of the four in a TunedModel if they are interested in comparing the models over a large set of their hyperparameters.

### Comparing the models
We simply pass the four models to the `models` argument of the `TunedModel` construct

````julia
tuned_model = TunedModel(
    models=[clf1, clf2, clf3, tuned_model_xg],
    tuning=Explicit(),
    resampling=CV(nfolds=5, rng=42),
    measure=cross_entropy,
);
````

Then wrapping our tuned model in a machine and fitting it.

````julia
mach = machine(tuned_model, X, y);
fit!(mach, verbosity=0);
````

````
┌ Warning: Layer with Float32 parameters got Float64 input.
│   The input will be converted, but any earlier layers may be very slow.
│   layer = Dense(4 => 5, relu)  # 25 parameters
│   summary(x) = "4×8 Matrix{Float64}"
└ @ Flux ~/.julia/packages/Flux/Wz6D4/src/layers/stateless.jl:60

````

Now let's see the history for more details on the performance for each of the models

````julia
history = report(mach).history
history_df = DataFrame(mlp = [x[:model] for x in history], measurement = [x[:measurement][1] for x in history])
sort!(history_df, [order(:measurement)])
````

```@raw html
<div><div style = "float: left;"><span>4×2 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">mlp</th><th style = "text-align: left;">measurement</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Probabilistic" style = "text-align: left;">Probabil…</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">BayesianLDA(method = gevd, …)</td><td style = "text-align: right;">0.0610826</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">RandomForestClassifier(max_depth = -1, …)</td><td style = "text-align: right;">0.106565</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">NeuralNetworkClassifier(builder = MLP(hidden = (5, 4), …), …)</td><td style = "text-align: right;">0.113266</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">ProbabilisticTunedModel(model = XGBoostClassifier(test = 1, …), …)</td><td style = "text-align: right;">0.221056</td></tr></tbody></table></div>
```

This is Occam's razor in practice.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

