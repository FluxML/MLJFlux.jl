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

    pred = MLJBase.predict(model, fitresult, images[1:5])

    model.epochs = 15
    MLJBase.update(model, 3, fitresult, cache, images, labels)
end

@testset "Image MNIST" begin
    using Flux.Data:MNIST

    images, labels = MNIST.images(), MNIST.labels()

    # Images here are of dimension 28x28
    # They need to be 28x28x1 according to the
    # convention.
    images = [reshape(image, 28, 28, 1) for image in images]
    labels = categorical(labels)

    function MLJFlux.fit(model::mnistclassifier, ip, op)
        dense_layers = ip[1:2] .- model.kernel1 .+ 1 .- model.kernel2 .+ 1 |> prod

        return Flux.Chain(Flux.Conv(model.kernel1, 1=>model.filters1),
                            Flux.Conv(model.kernel2, model.filters1=>model.filters2), vec,
                                Flux.Dense(dense_layers, op))
    end

    model = MLJFlux.ImageClassifier(builder=mnistclassifier((3,3), 2, (3,3), 1))

    fitresult, cache, report = MLJBase.fit(model, 3, images[1:500], labels[1:500])

end

true
