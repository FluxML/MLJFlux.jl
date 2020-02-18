module MLJFlux

import Flux
import MLJBase
import Base.==
using Base.Iterators: partition
using ProgressMeter
using CategoricalArrays
using Tables

include("core.jl")
include("regressor.jl")
include("classifier.jl")
include("image.jl")

### Package specific traits:
MLJBase.metadata_pkg.((NeuralNetworkRegressor, NeuralNetworkClassifier,ImageClassifier, MultivariateNeuralNetworkRegressor),
              name="MLJFlux",
              uuid="094fc8d1-fd35-5302-93ea-dabda2abf845",
              url="https://github.com/alan-turing-institute/MLJFlux.jl",
              julia=true,
              license="MIT")

export NeuralNetworkRegressor, MultivariateNeuralNetworkRegressor
export NeuralNetworkClassifier, ImageClassifier

end #module
