Most MLJFlux models support categorical features by learning enitity embeddings. By
wrapping such an MLJFlux model using [`EntityEmbedder`](@ref), the learned embeddings can
be used in MLJ pipelines as transformer elements. In particular, these embeddings can be
used for supervised models that are not neural networks and require features to be
`Continuous`. See the example below.

```@docs
MLJFlux.EntityEmbedder
```
