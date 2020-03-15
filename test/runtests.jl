# using Revise
using Test
using Tables
import MLJBase
import MLJFlux
using CategoricalArrays
import Flux
import Random.seed!
using Statistics
seed!(123)

# test equality of optimisers:
@test Flux.Momentum() == Flux.Momentum()
@test Flux.Momentum(0.1) != Flux.Momentum(0.2)

include("classifier.jl")
include("regressor.jl")