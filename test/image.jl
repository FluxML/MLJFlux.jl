## BASIC IMAGE TESTS GREY

Random.seed!(123)

mutable struct mynn <: MLJFlux.Builder
    kernel1
    kernel2
end

MLJFlux.build(model::mynn, ip, op, n_channels) =
        Flux.Chain(Flux.Conv(model.kernel1, n_channels=>2),
                   Flux.Conv(model.kernel2, 2=>1),
                   x->reshape(x, :, size(x)[end]),
                   Flux.Dense(16, op))

builder = mynn((2,2), (2,2))

# collection of gray images as a 4D array in WHCN format:
raw_images = rand(Float32, 6, 6, 1, 50);

# as a vector of Matrix{<:AbstractRGB}
images = coerce(raw_images, GrayImage);
@test scitype(images) == AbstractVector{GrayImage{6,6}}

labels = categorical(rand(1:5, 50));

@testset_accelerated "ImageClassifier basic tests" accel begin

    Random.seed!(123)

    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    acceleration=accel)

    fitresult, cache, report = MLJBase.fit(model, 3, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    model.epochs = 15
    MLJBase.update(model, 3, fitresult, cache, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    # try with batch_size > 1:
    model = MLJFlux.ImageClassifier(builder=builder, epochs=10, batch_size=2)
    fitresult, cache, report = MLJBase.fit(model, 3, images, labels);

    # tests update logic, etc (see test_utililites.jl):
    @test basictest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, 0.95, accel)

end


## MNIST IMAGES TEST

mutable struct MyConvBuilder <: MLJFlux.Builder end

using Flux.Data:MNIST

images, labels = MNIST.images(), MNIST.labels();

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

@testset_accelerated "Image MNIST" accel begin

    Random.seed!(123)

    model = MLJFlux.ImageClassifier(builder=MyConvBuilder())

    fitresult, cache, report =
        MLJBase.fit(model, 3, images[1:500], labels[1:500]);

    pred = mode.(MLJBase.predict(model, fitresult, images[501:600]));
    error = misclassification_rate(pred, labels[501:600])
    @test error < 0.1

end


## BASIC IMAGE TESTS COLOR

builder = mynn((2,2), (2,2))

# collection of color images as a 4D array in WHCN format:
raw_images = rand(Float32, 6, 6, 3, 50);

images = coerce(raw_images, ColorImage);
@test scitype(images) == AbstractVector{ColorImage{6,6}}

labels = categorical(rand(1:5, 50));

@testset_accelerated "ColorImages" accel begin

    Random.seed!(123)

    model = MLJFlux.ImageClassifier(builder=builder, epochs=10)

    fitresult, cache, report = MLJBase.fit(model, 3, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    model.epochs = 15
    MLJBase.update(model, 3, fitresult, cache, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    # try with batch_size > 1:
    model = MLJFlux.ImageClassifier(builder=builder, epochs=10, batch_size=2)
    fitresult, cache, report = MLJBase.fit(model, 3, images, labels);

    # tests update logic, etc (see test_utililites.jl):
    @test basictest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, 0.95, accel)
end

true
