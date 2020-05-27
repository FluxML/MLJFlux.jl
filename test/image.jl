mutable struct mynn <: MLJFlux.Builder
    kernel1
    kernel2
end

mutable struct mnistclassifier <: MLJFlux.Builder
    kernel1
    filters1
    kernel2
    filters2
end

@testset "ImageClassifier" begin

    MLJFlux.fit(model::mynn, ip, op) =
        Flux.Chain(Flux.Conv(model.kernel1, 1=>2),
                   Flux.Conv(model.kernel2, 2=>1),
                   vec,
                   Flux.Dense(16, op))

    model = MLJFlux.ImageClassifier(builder = mynn((2,2), (2,2)), epochs=10)

    # collection of gray images as a 4D array in WHCN format:
    raw_images = rand(6, 6, 1, 50);

    # as a vector of Matrix{<:AbstractRGB}
    images = coerce(raw_images, GrayImage);
    @test scitype(images) == AbstractVector{GrayImage{6,6}}

    labels = categorical(rand(1:5, 50));

    fitresult, cache, report = MLJBase.fit(model, 3, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    model.epochs = 15
    MLJBase.update(model, 3, fitresult, cache, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

end

@testset "Image MNIST" begin
    using Flux.Data:MNIST

    images, labels = MNIST.images(), MNIST.labels()

    labels = categorical(labels)

    function flatten(x::AbstractArray)
        return reshape(x, :, size(x)[end])
    end

    function MLJFlux.fit(model::mnistclassifier, ip, op)
        cnn_output_size = [3,3,32]	

        return Chain(
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_size), 10))
    end

    model = MLJFlux.ImageClassifier(builder=mnistclassifier((3,3), 2, (3,3), 1))

    fitresult, cache, report = MLJBase.fit(model, 3, images[1:500], labels[1:500])
    pred = MLJBase.predict(model, fitresult, images[1:5])

end

true
