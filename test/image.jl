@testset "ImageClassifier" begin
    mutable struct mynn <: MLJFlux.Builder
        kernel1
        kernel2
    end

    function MLJFlux.fit(model::mynn, ip, op)
        return Flux.Chain(Flux.Conv(model.kernel1, 1=>2), Flux.Conv(model.kernel2, 2=>1), vec, Flux.Dense(16, op))
    end

    model = MLJFlux.ImageClassifier(builder = mynn((2,2), (2,2)), epochs=10)
    img, l = [rand(6,6,1) for i=1:50], CategoricalArray(rand(1:5, 50))

    fitresult, cache, report = MLJBase.fit(model, 3, img, l)

    pred = MLJBase.predict(model, fitresult, img[1:5])
    
    model.epochs = 15
    MLJBase.update(model, 3, fitresult, cache, img, l)
end

true