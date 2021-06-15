function shape(model::ImageClassifier, X, y)
    levels = MLJModelInterface.classes(y[1])
    n_output = length(levels)
    n_input = size(X[1])

    if scitype(first(X)) <: GrayImage{A, B} where A where B
        n_channels = 1      # 1-D image
    else
        n_channels = 3      # 3-D color image
    end
    return (n_input, n_output, n_channels)
end

build(model::ImageClassifier, shape) =
    Flux.Chain(build(model.builder, shape...),
               model.finaliser)

fitresult(model::ImageClassifier, chain, y) =
    (chain, MLJModelInterface.classes(y[1]))

function MLJModelInterface.predict(model::ImageClassifier, fitresult, Xnew)
    chain, levels = fitresult
    X = reformat(Xnew)
    probs = vcat([chain(X[:,:,:,idx:idx])'
                  for idx in 1:last(size(X))]...)
    return MLJModelInterface.UnivariateFinite(levels, probs)
end

MLJModelInterface.metadata_model(ImageClassifier,
               input=AbstractVector{<:MLJModelInterface.Image},
               target=AbstractVector{<:Multiclass},
               path="MLJFlux.ImageClassifier",
               descr="A neural network model for making probabilistic "*
                     "predictions of a `GrayImage` target, "*
                     "given a table of `Continuous` features. ")
