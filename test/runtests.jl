using Test
using Tables
import MLJBase
import MLJFlux
using CategoricalArrays
import Flux
import Random
import Random.seed!
using Statistics
seed!(123)

@testset "core" begin
    include("core.jl")
end

@testset "regressor" begin
    include("regressor.jl")
end

@testset "classifier" begin
    include("classifier.jl")
end
