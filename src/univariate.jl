# This is just some experimental code
# to implement EntityEmbeddings for purely
# categorical features


using Flux

mutable struct EmbeddingMatrix
    e
    levels

    function EmbeddingMatrix(levels; dimension=4)
        return new(Dense(length(levels), dimension), levels)
    end

end

Flux.@treelike EmbeddingMatrix

function (embed::EmbeddingMatrix)(ip)
    return embed.e(Flux.onehot(ip, embed.levels))
end

mutable struct EntityEmbedding
    embeddingmatrix

    function EntityEmbedding(a...)
        return new(a)
    end
end

Flux.@treelike EntityEmbedding

function (embed::EntityEmbedding)(ip)
    return vcat((embed.embeddingmatrix[i](ip[i]) for i=1:length(ip))...)
end


#   Q1. How should this be called in the API?
#   nn = NeuralNetworkClassifier(builder=builder, optimiser = .., embeddingdimension = 5)
#
#
#
