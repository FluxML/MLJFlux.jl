using Test
using Tables
using MLJBase
import MLJFlux
using CategoricalArrays
using ColorTypes
using Flux
import Random
import Random.seed!
using Statistics
import StatsBase
using MLJModelInterface.ScientificTypes

using ComputationalResources
using ComputationalResources: CPU1, CUDALibs

const RESOURCES = Any[CPU1(), CUDALibs()]
# const RESOURCES = Any[CPU1(), CUDALibs()]
const EXCLUDED_RESOURCE_TYPES = Any[]
# const EXCLUDED_RESOURCE_TYPES = Any[CUDALibs,]

# alternative version of Short builder with no dropout; see
# https://github.com/FluxML/Flux.jl/issues/1372
mutable struct Short2 <: MLJFlux.Builder
    n_hidden::Int     # if zero use geometric mean of input/output
    σ
end
Short2(; n_hidden=0, σ=Flux.sigmoid) = Short2(n_hidden, σ)
function MLJFlux.build(builder::Short2, n, m)
    n_hidden =
        builder.n_hidden == 0 ? round(Int, sqrt(n*m)) : builder.n_hidden
    return Flux.Chain(Flux.Dense(n, n_hidden, builder.σ),
                       Flux.Dense(n_hidden, m))
end

seed!(123)

include("test_utils.jl")

@testset "core" begin
    include("core.jl")
end

@testset "regressor" begin
    include("regressor.jl")
end

@testset "classifier" begin
    include("classifier.jl")
end

@testset "image" begin
    include("image.jl")
end
