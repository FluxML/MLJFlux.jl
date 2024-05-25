### Adding new models to MLJFlux 

This section assumes familiarity
with the [MLJ model
API](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/)

If one subtypes a new model type as either
`MLJFlux.MLJFluxProbabilistic` or `MLJFlux.MLJFluxDeterministic`, then
instead of defining new methods for `MLJModelInterface.fit` and
`MLJModelInterface.update` one can make use of fallbacks by
implementing the lower level methods `shape`, `build`, and
`fitresult`. See the [classifier source code](https://github.com/FluxML/MLJFlux.jl/blob/dev/src/classifier.jl) for
an example.

One still needs to implement a new `predict` method.