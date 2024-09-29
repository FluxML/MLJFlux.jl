using Test
using Tables
using MLJBase
import MLJFlux
using CategoricalArrays
using MLJBase
using ColorTypes
using Flux
import Random
import Random.seed!
using Statistics
import StatsBase
using StableRNGs
using CUDA, cuDNN
import StatisticalMeasures
import Optimisers
import Logging

using ComputationalResources
using ComputationalResources: CPU1, CUDALibs

const RESOURCES = Any[CPU1(), CUDALibs()]
EXCLUDED_RESOURCE_TYPES = Any[]

MLJFlux.gpu_isdead() && push!(EXCLUDED_RESOURCE_TYPES, CUDALibs)

@info "MLJFlux supports these computational resources:\n$RESOURCES"
@info "Current test run to exclude resources with "*
    "these types, as unavailable:\n$EXCLUDED_RESOURCE_TYPES\n"*
    "Excluded tests marked as \"broken\"."

seed!(123)

include("test_utils.jl")

# enable conditional testing of modules by providing test_args
# e.g. `Pkg.test("MLJBase", test_args=["misc"])`
RUN_ALL_TESTS = isempty(ARGS)
macro conditional_testset(name, expr)
    name = string(name)
    esc(quote
        if RUN_ALL_TESTS || $name in ARGS
            @testset $name $expr
        end
    end)
end

@conditional_testset "core" begin
    include("core.jl")
end

@conditional_testset "builders" begin
    include("builders.jl")
end

@conditional_testset "metalhead" begin
    include("metalhead.jl")
end

@conditional_testset "mlj_model_interface" begin
    include("mlj_model_interface.jl")
end

@conditional_testset "regressor" begin
    include("regressor.jl")
end

@conditional_testset "classifier" begin
    include("classifier.jl")
end

@conditional_testset "image" begin
    include("image.jl")
end

@conditional_testset "integration" begin
    include("integration.jl")
end

@conditional_testset "entity embedding" begin
    include("entity_embedding.jl")
    include("entity_embedding_utils.jl")
    include("encoders.jl")
end