module MLJFlux

export CUDALibs, CPU1
import Flux
using MLJModelInterface
using MLJModelInterface.ScientificTypesBase
import Base.==
using ProgressMeter
using CategoricalArrays
using Tables
using Statistics
using ColorTypes
using ComputationalResources
using Random
import Metalhead
import Optimisers

include("utilities.jl")
const MMI = MLJModelInterface

include("encoders.jl")
include("entity_embedding.jl")
include("builders.jl")
include("metalhead.jl")
include("types.jl")
include("core.jl")
include("regressor.jl")
include("classifier.jl")
include("image.jl")
include("fit_utils.jl")
include("entity_embedding_utils.jl")
include("mlj_model_interface.jl")

export NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor
export NeuralNetworkClassifier, NeuralNetworkBinaryClassifier, ImageClassifier
export CUDALibs, CPU1

include("deprecated.jl")
end # module
