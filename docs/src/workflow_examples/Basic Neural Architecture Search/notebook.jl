# # Neural Architecture Search with MLJFlux

# Neural Architecture Search is (NAS) is an instance of hyperparameter tuning concerned
# with tuning model hyperparameters defining the architecture itself. Although it's
# typically performed with sophisticated search algorithms for efficiency, in this example
# we will be using a simple random search.

using Pkg     #!md
Pkg.activate(@__DIR__);     #!md
Pkg.instantiate();     #!md

# **Julia version** is assumed to be 1.10.* This tutorial is available as a Jupyter
# notebook or julia script
# [here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/workflow_examples/architecture_search).

# ### Basic Imports

using MLJ               # Has MLJFlux models
using Flux              # For more flexibility
using RDatasets: RDatasets        # Dataset source
using DataFrames        # To view tuning results in a table
import Optimisers       # native Flux.jl optimisers no longer supported

# ### Loading and Splitting the Data

iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true, rng = 123);
X = Float32.(X);      # To be compatible with type of network network parameters
first(X, 5)


# ### Instantiating the model

# Now let's construct our model. This follows a similar setup the one followed in the
# [Quick Start](../../index.md#Quick-Start).

NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg = "MLJFlux"
clf = NeuralNetworkClassifier(
    builder = MLJFlux.MLP(; hidden = (1, 1, 1), σ = Flux.relu),
    optimiser = Optimisers.ADAM(0.01),
    batch_size = 8,
    epochs = 10,
    rng = 42,
)


# ### Generating Network Architectures

# We know that the MLP builder takes a tuple of the form $(z_1, z_2, ..., z_k)$ to define
# a network with $k$ hidden layers and where the ith layer has $z_i$ neurons. We will
# proceed by defining a function that can generate all possible networks with a specific
# number of hidden layers, a minimum and maximum number of neurons per layer and
# increments to consider for the number of neurons.

function generate_networks(
    ;min_neurons::Int,
    max_neurons::Int,
    neuron_step::Int,
    num_layers::Int,
    )
    ## Define the range of neurons
    neuron_range = min_neurons:neuron_step:max_neurons

    ## Empty list to store the network configurations
    networks = Vector{Tuple{Vararg{Int, num_layers}}}()

    ## Recursive helper function to generate all combinations of tuples
    function generate_tuple(current_layers, remaining_layers)
        if remaining_layers > 0
            for n in neuron_range
                ## current_layers =[] then current_layers=[(min_neurons)],
                ## [(min_neurons+neuron_step)], [(min_neurons+2*neuron_step)],...
                ## for each of these we call generate_layers again which appends
                ## the n combinations for each one of them
                generate_tuple(vcat(current_layers, [n]), remaining_layers - 1)
            end
        else
            ## in the base case, no more layers to "recurse on"
            ## and we just append the current_layers as a tuple
            push!(networks, tuple(current_layers...))
        end
    end

    ## Generate networks for the given number of layers
    generate_tuple([], num_layers)

    return networks
end


# Now let's generate an array of all possible neural networks with three hidden layers and
# number of neurons per layer ∈ [1,64] with a step of 4
networks_space =
    generate_networks(
        min_neurons = 1,
        max_neurons = 64,
        neuron_step = 4,
        num_layers = 3,
    )

networks_space[1:5]

# ### Wrapping the Model for Tuning

# Let's use this array to define the range of hyperparameters and pass it along with the
# model to the `TunedModel` constructor.
r1 = range(clf, :(builder.hidden), values = networks_space)

tuned_clf = TunedModel(
    model = clf,
    tuning = RandomSearch(),
    resampling = CV(nfolds = 4, rng = 42),
    range = [r1],
    measure = cross_entropy,
    n = 100,             # searching over 100 random samples are enough
);

# ### Performing the Search

# Similar to the last workflow example, all we need now is to fit our model and the search
# will take place automatically:
mach = machine(tuned_clf, X, y);
fit!(mach, verbosity = 0);
fitted_params(mach).best_model

# ### Analyzing the Search Results

# Let's analyze the search results by converting the history array to a dataframe and
# viewing it:
history = report(mach).history
history_df = DataFrame(
    mlp = [x[:model].builder for x in history],
    measurement = [x[:measurement][1] for x in history],
)
first(sort!(history_df, [order(:measurement)]), 10)
