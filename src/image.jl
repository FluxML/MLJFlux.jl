function shape(model::ImageClassifier, X, y)
    levels = CategoricalArrays.levels(y)
    n_output = length(levels)
    n_input = size(X[1])

    if scitype(first(X)) <: GrayImage{A, B} where A where B
        n_channels = 1      # 1-D image
    else
        n_channels = 3      # 3-D color image
    end
    return (n_input, n_output, n_channels)
end
is_embedding_enabled(::ImageClassifier) = false


build(model::ImageClassifier, rng, shape) =
    Flux.Chain(build(model.builder, rng, shape...),
               model.finaliser)

fitresult(model::ImageClassifier, chain, y, ::Any, ::Any) =
    (chain, levels(y))

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
               path="MLJFlux.ImageClassifier")
