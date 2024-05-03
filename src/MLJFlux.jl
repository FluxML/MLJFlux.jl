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
const MMI=MLJModelInterface

include("builders.jl")
include("metalhead.jl")
include("types.jl")
include("core.jl")
include("regressor.jl")
include("classifier.jl")
include("image.jl")
include("mlj_model_interface.jl")

export NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor
export NeuralNetworkClassifier, ImageClassifier
export CUDALibs, CPU1



end #module
