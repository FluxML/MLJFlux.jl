Most MLJFlux models support categorical features by learning enitity embeddings, as
described in article, "Entity Embeddings of Categorical Variables" by Cheng Guo, Felix
Berkhahn. By wrapping such an MLJFlux model using [`EntityEmbedder`](@ref), the learned
embeddings can be used in MLJ pipelines as transformer elements. See the example below.

```@docs
MLJFlux.EntityEmbedder
```
