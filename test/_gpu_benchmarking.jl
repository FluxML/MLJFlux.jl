# This stand-alone file is for benchmarking CPU/GPU on MNIST dataset

using MLJFlux, BenchmarkTools, Flux, ComputationalResources

mutable struct MyConvBuilder <: MLJFlux.Builder end

using Flux.Data:MNIST

N = 500
images, labels = MNIST.images()[1:N], MNIST.labels()[1:N];

labels = categorical(labels);

function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

function MLJFlux.build(builder::MyConvBuilder, n_in, n_out, n_channels)
    cnn_output_size = [3,3,32]

    return Chain(
        Conv((3, 3), n_channels=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_size), n_out))
end

basedir = joinpath(dirname(pathof(MLJFlux)), "..", "test")
include(joinpath(basedir, "test_utils.jl"))

# CPU
model = MLJFlux.ImageClassifier(builder=MyConvBuilder(),
                                batch_size=50,
                                acceleration=CPU1());
@btime MLJBase.fit($model, 0, $images, $labels);
# 16.877 s (7401673 allocations: 8.15 GiB) batch_size = 1 
#  8.816 s (409635 allocations: 3.15 GiB) batch_size = 50

# GPU
model.acceleration = CUDALibs()
@btime MLJBase.fit($model, 0, $images, $labels);
# 30.214 s (46970158 allocations: 2.37 GiB) batch_size = 1
# 600.534 ms (960560 allocations: 52.82 MiB) batch_size = 50


