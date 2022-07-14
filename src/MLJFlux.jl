module MLJFlux

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

include("utilities.jl")
include("penalizers.jl")
include("core.jl")
include("builders.jl")
include("metalhead.jl")
include("types.jl")
include("regressor.jl")
include("classifier.jl")
include("image.jl")
include("mlj_model_interface.jl")

### Package specific model traits:
MLJModelInterface.metadata_pkg.((NeuralNetworkRegressor,
                                 MultitargetNeuralNetworkRegressor,
                                 NeuralNetworkClassifier,
                                 ImageClassifier),
              name="MLJFlux",
              uuid="094fc8d1-fd35-5302-93ea-dabda2abf845",
              url="https://github.com/alan-turing-institute/MLJFlux.jl",
              julia=true,
              license="MIT")

export NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor
export NeuralNetworkClassifier, ImageClassifier
export CUDALibs, CPU1

end #module
