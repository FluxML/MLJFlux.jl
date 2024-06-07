# # Model Composition with MLJFlux

# In this workflow example, we see how MLJFlux enables composing MLJ models with MLJFlux models. We will assume a 
# class imbalance setting and wrap an oversampler with a deep learning model from MLJFlux.

using Pkg     #src
Pkg.activate(@__DIR__);     #src
Pkg.instantiate();     #src

# **Julia version** is assumed to be 1.10.*

# ### Basic Imports

using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
import RDatasets        # Dataset source
import Random           # To create imbalance
import Imbalance        # To solve the imbalance

# ### Loading and Splitting the Data

iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng=123);
X = Float32.(X);      # To be compatible with type of network network parameters

# To simulate an imbalanced dataset, we will take a random sample:
Random.seed!(803429)
subset_indices = rand(1:size(X, 1), 100)
X, y = X[subset_indices, :], y[subset_indices]
Imbalance.checkbalance(y)        



# ### Instantiating the model

# Let's load `BorderlineSMOTE1` to oversample the data and `Standardizer` to standardize it.
BorderlineSMOTE1 = @load BorderlineSMOTE1 pkg=Imbalance verbosity=0
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
Standardizer = @load Standardizer pkg=MLJModels

clf = NeuralNetworkClassifier(
    builder=MLJFlux.MLP(; hidden=(5,4), σ=Flux.relu),
    optimiser=Flux.ADAM(0.01),
    batch_size=8,
    epochs=50, 
    rng=42
    )

# First we wrap the oversampler with the neural network via the `BalancedModel` construct. This comes from `MLJBalancing`
# And allows combining resampling methods with MLJ models in a sequential pipeline.
oversampler = BorderlineSMOTE1(k=5, ratios=1.0, rng=42)
balanced_model = BalancedModel(model=clf, balancer1=oversampler)
standarizer = Standardizer()

# Now let's compose the balanced model with a standardizer.
pipeline = standarizer |> balanced_model
# By this, any training data will be standardized then oversampled then passed to the model. Meanwhile,
# for inference, the standardizer will automatically use the training set's mean and std and the oversampler
# will be transparent.


# ### Training the Composed Model
# It's indistinguishable from training a single model.

mach = machine(pipeline, X, y)
fit!(mach)
cv=CV(nfolds=5)
evaluate!(mach, resampling=cv, measure=accuracy) 



using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=false) #src
Literate.notebook(@__FILE__, @__DIR__, execute=true) #src
