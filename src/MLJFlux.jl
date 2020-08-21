module MLJFlux

import Flux
import MLJModelInterface
using ScientificTypes
import Base.==
using ProgressMeter
using CategoricalArrays
using Tables
using Statistics
using ColorTypes
using ComputationalResources

include("core.jl")
include("regressor.jl")
include("classifier.jl")
include("image.jl")

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

end #module
