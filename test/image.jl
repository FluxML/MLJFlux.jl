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

    function MLJFlux.fit(model::mynn, ip, op)
        return Flux.Chain(Flux.Conv(model.kernel1, 1=>2), Flux.Conv(model.kernel2, 2=>1), vec, Flux.Dense(16, op))
    end

    model = MLJFlux.ImageClassifier(builder = mynn((2,2), (2,2)), epochs=10)
    img, l = [Gray.(rand(6,6)) for i=1:50], CategoricalArray(rand(1:5, 50))

    fitresult, cache, report = MLJBase.fit(model, 3, img, l)

    pred = MLJBase.predict(model, fitresult, img[1:5])
  
    model.epochs = 15
    MLJBase.update(model, 3, fitresult, cache, img, l)

    pred = MLJBase.predict(model, fitresult, img[1:5])
end

@testset "Image MNIST" begin
    using Flux.Data:MNIST

    images, labels = MNIST.images(), MNIST.labels()

    # Images here are of dimension 28x28
    # They need to be 28x28x1 according to the 
    # convention.
    images = [Gray.(reshape(image, 28, 28)) for image in images]
    labels = categorical(labels)

    function MLJFlux.fit(model::mnistclassifier, ip, op)
        dense_layers = ip[1:2] .- model.kernel1 .+ 1 .- model.kernel2 .+ 1 |> prod

        return Flux.Chain(Flux.Conv(model.kernel1, 1=>model.filters1), 
                            Flux.Conv(model.kernel2, model.filters1=>model.filters2), vec,
                                Flux.Dense(dense_layers, op))
    end

    model = MLJFlux.ImageClassifier(builder=mnistclassifier((3,3), 2, (3,3), 1))

    fitresult, cache, report = MLJBase.fit(model, 3, images[1:500], labels[1:500])
    pred = MLJBase.predict(model, fitresult, images[1:5])

end
true