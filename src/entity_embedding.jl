"""
A layer that implements entity embedding layers as presented in 'Entity Embeddings of
 Categorical Variables by Cheng Guo, Felix Berkhahn'. Expects a matrix of dimensions (numfeats, batchsize)
 and applies entity embeddings to each specified categorical feature. Other features will be left as is.

# Arguments
- `entityprops`: a vector of named tuples each of the form `(index=..., levels=..., newdim=...)` to 
    specify the feature index, the number of levels and the desired embeddings dimensionality for selected features of the input.
- `numfeats`: the number of features in the input.

# Example
```julia
# Prepare a batch of four features where the 2nd and the 4th are categorical
batch = [
    0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1;
    1 2 3 4 5 6 7 8 9 10;
    0.9 0.1 0.4 0.5 0.3 0.7 0.8 0.9 1.0 1.1
    1 1 2 2 1 1 2 2 1 1;
]

entityprops = [
    (index=2, levels=10, newdim=2),
    (index=4, levels=2, newdim=1)
]
numfeats = 4

# Run it through the categorical embedding layer
embedder = CategoricalEmbedder(entityprops, 4)
julia> output = embedder(batch)
5×10 Matrix{Float64}:
  0.2        0.3        0.4        0.5       …   0.8        0.9         1.0        1.1
 -1.27129   -0.417667  -1.40326   -0.695701      0.371741   1.69952    -1.40034   -2.04078
 -0.166796   0.657619  -0.659249  -0.337757     -0.717179  -0.0176273  -1.2817    -0.0372752
  0.9        0.1        0.4        0.5           0.8        0.9         1.0        1.1
 -0.847354  -0.847354  -1.66261   -1.66261      -1.66261   -1.66261    -0.847354  -0.847354
```
"""

# 1. Define layer struct to hold parameters
struct CategoricalEmbedder{A1 <: AbstractVector, A2 <: AbstractVector, I <: Integer}
    embedders::A1
    modifiers::A2
    numfeats::I
end

# 2. Define the forward pass (i.e., calling an instance of the layer)
(m::CategoricalEmbedder)(x) = vcat([ m.embedders[i](m.modifiers[i](x, i)) for i in 1:m.numfeats]...)

# 3. Define the constructor which initializes the parameters and returns the instance
function CategoricalEmbedder(entityprops, numfeats; init=Flux.randn32)
    embedders = []
    modifiers = []

    cat_inds = [entityprop.index for entityprop in entityprops]
    levels_per_feat = [entityprop.levels for entityprop in entityprops]
    newdims = [entityprop.newdim for entityprop in entityprops]
    
    c = 1
    for i in 1:numfeats
        if i in cat_inds
            push!(embedders, Flux.Embedding(levels_per_feat[c] => newdims[c], init=init))
            push!(modifiers, (x, i) -> Int.(x[i, :]))
            c += 1
        else
            push!(embedders, feat->feat)
            push!(modifiers, (x, i) -> x[i:i, :])
        end
    end

    CategoricalEmbedder(embedders, modifiers, numfeats)
end

# 4. Register it as layer with Flux
Flux.@layer CategoricalEmbedder