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
const EXCLUDED_RESOURCE_TYPES = Any[CUDALibs,]

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
