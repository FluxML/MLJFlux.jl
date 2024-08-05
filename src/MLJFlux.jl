module MLJFlux

export CUDALibs, CPU1

using MLJModelInterface
using MLJModelInterface.ScientificTypesBase
using ScientificTypes: schema, Finite
import Base.==
using ProgressMeter
using CategoricalArrays
using Tables
using Statistics
using ColorTypes
using ComputationalResources
using Random

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
include("mlj_model_interface.jl")

export NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor
export NeuralNetworkClassifier, NeuralNetworkBinaryClassifier, ImageClassifier
export CUDALibs, CPU1
export EntityEmbedder

include("deprecated.jl")
end
