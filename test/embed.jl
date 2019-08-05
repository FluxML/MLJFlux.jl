using MLJFlux
using MLJ
using MLJBase
using CategoricalArrays
using Flux
using Test

n = 5
N = 1000

v =rand(1:n, N)
y = float.(v)

x1 = categorical(v)
levels!(x1, collect(1:n))
@assert all(x1 .== MLJBase.int(x1))

X = (x1 = x1, )

struct testnn <: MLJFlux.Builder
    d
end

function  MLJFlux.fit(model::testnn, ip, op)
    return Chain(identity)
end

nn = testnn(5)
nnmodel = NeuralNetworkRegressor(builder=nn, embedding_choice=:entity_embedding, optimiser=ADAM(0.0001), n=120, embeddingdimension=1)

fitresult, cache, report = MLJBase.fit(nnmodel, 2, X, y)

chain = fitresult[1];

embeddings = chain.layers[1]

embedding_values = []
for embedding_matrix in embeddings.embeddingmatrix
    push!(embedding_values, embedding_matrix.e)
end

@test sum(embedding_values[1].W .+ embedding_values[1].b .- [1; 2; 3; 4; 5]) ≈ 0
