"""
A file containing functions or constants used in the `fit` and `update` methods in `mlj_model_interface.jl` for setups supporting entity embeddings
"""
is_embedding_enabled(model) = false

# function to set default new embedding dimension 
function set_default_new_embedding_dim(numlevels)
    # Set default to the minimum of num_levels-1 and 10
    return min(numlevels - 1, 10)
end

MISMATCH_INDS(wrong_feats) =
    "Features $(join(wrong_feats, ", ")) were specified in embedding_dims hyperparameter but were not recognized as categorical variables because their scitypes are not `Multiclass` or `OrderedFactor`."
function check_mismatch_in_cat_feats(featnames, cat_inds, specified_featinds)
    wrong_feats = [featnames[i] for i in specified_featinds if !(i in cat_inds)]
    length(wrong_feats) > 0 && throw(ArgumentError(MISMATCH_INDS(wrong_feats)))
end

# function to set new embedding dimensions
function set_new_embedding_dims(featnames, cat_inds, num_levels, embedding_dims)
    specified_featnames = keys(embedding_dims)
    specified_featinds =
        [i for i in 1:length(featnames) if featnames[i] in specified_featnames]
    check_mismatch_in_cat_feats(featnames, cat_inds, specified_featinds)
    catind2numlevels = Dict(zip(cat_inds, num_levels))
    # for each value of embedding dim if float then multiply it by the number of levels
    for featname in specified_featnames
        if embedding_dims[featname] isa AbstractFloat
            embedding_dims[featname] = ceil(
                Int,
                embedding_dims[featname] *
                catind2numlevels[findfirst(x -> x == featname, featnames)],
            )
        end
    end
    newdims = [
        (cat_ind in specified_featinds) ? embedding_dims[featnames[cat_ind]] :
        set_default_new_embedding_dim(num_levels[i]) for
        (i, cat_ind) in enumerate(cat_inds)
    ]
    return newdims
end


"""
**Private Method**

Returns the indices of the categorical columns in the table `X`.
"""
function get_cat_inds(X)
    # if input is a matrix; conclude no categorical columns
    Tables.istable(X) || return Int[]
    types = [
        scitype(Tables.getcolumn(X, name)[1]) for
        name in Tables.schema(Tables.columns(X)).names
    ]
    cat_inds = findall(x -> x <: Finite, types)
    return cat_inds
end

"""
**Private Method**

Returns the number of levels in each categorical column in the table `X`.
"""
function get_num_levels(X, cat_inds)
    num_levels = []
    for i in cat_inds
        num_levels =
            push!(num_levels, length(levels(Tables.getcolumn(Tables.columns(X), i))))
    end
    return num_levels
end

# A function to prepare the inputs for entity embeddings layer
function prepare_entityembs(X, featnames, cat_inds, embedding_dims)
    # 1. Construct entityprops
    numfeats = length(featnames)
    num_levels = get_num_levels(X, cat_inds)
    newdims = set_new_embedding_dims(featnames, cat_inds, num_levels, embedding_dims)
    entityprops = [
        (index = cat_inds[i], levels = num_levels[i], newdim = newdims[i]) for
        i in eachindex(cat_inds)
    ]
    # 2. Compute entityemb_output_dim
    sum_newdims = length(newdims) == 0 ? 0 : sum(newdims)
    entityemb_output_dim = sum_newdims + numfeats - length(cat_inds)
    return entityprops, entityemb_output_dim
end

# A function to construct model chain including entity embeddings as the first layer
function construct_model_chain_with_entityembs(
    model,
    rng,
    shape,
    move,
    entityprops,
    entityemb_output_dim,
)
    chain = try
        Flux.Chain(
            EntityEmbedder(entityprops, shape[1]; init = Flux.glorot_uniform(rng)),
            build(model, rng, (entityemb_output_dim, shape[2])),
        ) |> move
    catch ex
        @error ERR_BUILDER
        rethrow()
    end
    return chain
end


# A function that given a model chain, returns a dictionary of embedding matrices
function get_embedding_matrices(chain, cat_inds, featnames)
    embedder_layer = chain.layers[1]
    embedding_matrices = Dict{Symbol, Matrix{Float32}}()
    for cat_ind in cat_inds
        featname = featnames[cat_ind]
        matrix = Flux.params(embedder_layer.embedders[cat_ind])[1]
        embedding_matrices[featname] = matrix
    end
    return embedding_matrices
end



# Transformer for entity-enabled models
function MLJModelInterface.transform(
    transformer::MLJFluxModel,
    fitresult,
    Xnew,
)
    # if it doesn't have the property its not an entity-enabled model
    hasproperty(transformer, :embedding_dims) || return Xnew
    ordinal_mappings, embedding_matrices = fitresult[3:4]
    Xnew = ordinal_encoder_transform(Xnew, ordinal_mappings)
    Xnew_transf = embedding_transform(Xnew, embedding_matrices)
    return Xnew_transf
end