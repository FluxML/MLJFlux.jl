# # HELPERS

function make_images(rng; n_classes=33, n_images=50, color=false, noise=0.05)
    n_channels = color ? 3 : 1
    image_bag = map(1:n_classes) do _
        rand(stable_rng, Float32, 6, 6, n_channels)
    end
    labels = rand(stable_rng, 1:3, n_images)
    images = map(labels) do j
        image_bag[j] + noise*rand(stable_rng, Float32, 6, 6, n_channels)
    end
    T = color ? ColorImage : GrayImage
    X = coerce(cat(images...; dims=4), T)
    y = coerce(labels, Multiclass)
    return X, y
end 
    

# # BASIC IMAGE TESTS GREY

Random.seed!(123)
stable_rng = StableRNGs.StableRNG(123)

mutable struct MyNeuralNetwork <: MLJFlux.Builder
    kernel1
    kernel2
end

function MLJFlux.build(model::MyNeuralNetwork, rng, ip, op, n_channels)
    init = Flux.glorot_uniform(rng)
    Flux.Chain(
        Flux.Conv(model.kernel1, n_channels=>2, init=init),
        Flux.Conv(model.kernel2, 2=>1, init=init),
        x->reshape(x, :, size(x)[end]),
        Flux.Dense(16, op, init=init))
end

builder = MyNeuralNetwork((2,2), (2,2))
images, labels = make_images(stable_rng)
losses = []

@testset_accelerated "ImageClassifier basic tests" accel begin

    Random.seed!(123)
    stable_rng = StableRNGs.StableRNG(123)

    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    acceleration=accel,
                                    rng=stable_rng)

    fitresult, cache, _report = MLJBase.fit(model, 0, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    model.epochs = 15
    MLJBase.update(model, 0, fitresult, cache, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    # try with batch_size > 1:
    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    batch_size=2,
                                    acceleration=accel,
                                    rng=stable_rng)
    model.optimiser.eta = 0.005
    @time fitresult, cache, _report = MLJBase.fit(model, 0, images, labels);
    first_last_training_loss = _report[1][[1, end]]
    push!(losses, first_last_training_loss[2])
#    @show first_last_training_loss

    # tests update logic, etc (see test_utililites.jl):
    @test basictest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, 0.95, accel)

    @test optimisertest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, accel)

end

# check different resources (CPU1, CUDALibs) give about the same loss:
reference = losses[1]
@info "Losses for each computational resource: $losses"
@test all(x->abs(x - reference)/reference < 5e-4, losses[2:end])


## MNIST IMAGES TEST

mutable struct MyConvBuilder <: MLJFlux.Builder end

using MLDatasets

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
images, labels = MNIST.traindata()
images = coerce(images, GrayImage);
labels = categorical(labels);

function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

function MLJFlux.build(builder::MyConvBuilder, rng, n_in, n_out, n_channels)
    cnn_output_size = [3,3,32]
    init = Flux.glorot_uniform(rng)
    return Chain(
        Conv((3, 3), n_channels=>16, pad=(1,1), relu, init=init),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), relu, init=init),
        MaxPool((2,2)),
        Conv((3, 3), 32=>32, pad=(1,1), relu, init=init),
        MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_size), n_out, init=init))
end

losses = []

@testset_accelerated "Image MNIST" accel begin

    Random.seed!(123)
    stable_rng = StableRNGs.StableRNG(123)

    model = MLJFlux.ImageClassifier(builder=MyConvBuilder(),
                                    acceleration=accel,
                                    batch_size=50,
                                    rng=stable_rng)

    @time fitresult, cache, _report =
        MLJBase.fit(model, 0, images[1:500], labels[1:500]);
    first_last_training_loss = _report[1][[1, end]]
    push!(losses, first_last_training_loss[2])
#    @show first_last_training_loss

    pred = mode.(MLJBase.predict(model, fitresult, images[501:600]));
    error = misclassification_rate(pred, labels[501:600])
    @test error < 0.2

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@info "Losses for each computational resource: $losses"
@test all(x->abs(x - reference)/reference < 0.05, losses[2:end])


## BASIC IMAGE TESTS COLOR

builder = MyNeuralNetwork((2,2), (2,2))
images, labels = make_images(stable_rng, color=true)
losses = []

@testset_accelerated "ColorImages" accel begin

    Random.seed!(123)
    stable_rng = StableRNGs.StableRNG(123)

    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    acceleration=accel,
                                    rng=stable_rng)

    # tests update logic, etc (see test_utililites.jl):
    @test basictest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, 0.95, accel)

    @time fitresult, cache, _report = MLJBase.fit(model, 0, images, labels)
    pred = MLJBase.predict(model, fitresult, images[1:6])
    first_last_training_loss = _report[1][[1, end]]
    push!(losses, first_last_training_loss[2])
#    @show first_last_training_loss

    # try with batch_size > 1:
    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    batch_size=2,
                                    acceleration=accel,
                                    rng=stable_rng)
    fitresult, cache, _report = MLJBase.fit(model, 0, images, labels);

    @test optimisertest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, accel)

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@info "Losses for each computational resource: $losses"
@test all(x->abs(x - reference)/reference < 1e-5, losses[2:end])

true
