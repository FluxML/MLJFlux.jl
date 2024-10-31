"""
File containing ordinal encoder and entity embedding encoder. Borrows code from the MLJTransforms package.
"""

### Ordinal Encoder
"""
**Private Method**

Fits an ordinal encoder to the table `X`, using only the columns with indices in `featinds`.

Returns a dictionary mapping each column index to a dictionary mapping each level in that column to an integer.
"""
function ordinal_encoder_fit(X; featinds)
    # 1. Define mapping per column per level dictionary
    mapping_matrix = Dict()

    # 2. Use feature mapper to compute the mapping of each level in each column
    for i in featinds
        feat_col = Tables.getcolumn(Tables.columns(X), i)
        feat_levels = levels(feat_col)
        # Check if feat levels is already ordinal encoded in which case we skip
        (Set([Float32(i) for i in 1:length(feat_levels)]) == Set(feat_levels)) && continue
        # Compute the dict using the given feature_mapper function
        mapping_matrix[i] =
            Dict{eltype(feat_levels), Float32}(
                value => Float32(index) for (index, value) in enumerate(feat_levels)
            )
    end
    return mapping_matrix
end

"""
**Private Method**

Checks that all levels in `test_levels` are also in `train_levels`. If not, throws an error.
"""
function check_unkown_levels(train_levels, test_levels)
    # test levels must be a subset of train levels
    if !issubset(test_levels, train_levels)
        # get the levels in test that are not in train
        lost_levels = setdiff(test_levels, train_levels)
        error(
            "While transforming, found novel levels for the column: $(lost_levels) that were not seen while training.",
        )
    end
end

"""
**Private Method**

Transforms the table `X` using the ordinal encoder defined by `mapping_matrix`.

Returns a new table with the same column names as `X`, but with categorical columns replaced by integer columns.
"""
function ordinal_encoder_transform(X, mapping_matrix)
    isnothing(mapping_matrix) && return X
    isempty(mapping_matrix) && return X
    feat_names = Tables.schema(X).names
    numfeats = length(feat_names)
    new_feats = []
    for ind in 1:numfeats
        col = Tables.getcolumn(Tables.columns(X), ind)

        # Create the transformation function for each column
        if ind in keys(mapping_matrix)
            train_levels = keys(mapping_matrix[ind])
            test_levels = levels(col)
            check_unkown_levels(train_levels, test_levels)
            level2scalar = mapping_matrix[ind]
            new_col = recode(col, level2scalar...)
            push!(new_feats, new_col)
        else
            push!(new_feats, col)
        end
    end

    transformed_X = NamedTuple{tuple(feat_names...)}(tuple(new_feats)...)
    # Attempt to preserve table type
    transformed_X = Tables.materializer(X)(transformed_X)
    return transformed_X
end

"""
**Private Method**

Combine ordinal_encoder_fit and ordinal_encoder_transform and return both X and ordinal_mappings
"""
function ordinal_encoder_fit_transform(X; featinds)
    ordinal_mappings = ordinal_encoder_fit(X; featinds = featinds)
    return ordinal_encoder_transform(X, ordinal_mappings), ordinal_mappings
end



## Entity Embedding Encoder (assuming precomputed weights)
"""
**Private method.**

Function to generate new feature names: feat_name_0, feat_name_1,..., feat_name_n
"""
function generate_new_feat_names(feat_name, num_inds, existing_names)
    conflict = true        # will be kept true as long as there is a conflict
    count = 1            # number of conflicts+1 = number of underscores

    new_column_names = []
    while conflict
        suffix = repeat("_", count)
        new_column_names = [Symbol("$(feat_name)$(suffix)$i") for i in 1:num_inds]
        conflict = any(name -> name in existing_names, new_column_names)
        count += 1
    end
    return new_column_names
end


"""
Given X and a dict of mapping_matrices that map each categorical column to a matrix, use the matrix to transform
each level in each categorical columns using the columns of the matrix.

This is used with the embedding matrices of the entity embedding layer in entity enabled models to implement entity embeddings.
"""
function embedding_transform(X, mapping_matrices)
    (isempty(mapping_matrices)) && return X
    feat_names = Tables.schema(X).names
    new_feat_names = Symbol[]
    new_cols = []
    for feat_name in feat_names
        col = Tables.getcolumn(Tables.columns(X), feat_name)
        # Create the transformation function for each column
        if feat_name in keys(mapping_matrices)
            level2vector = mapping_matrices[feat_name]
            new_multi_col = map(x -> level2vector[:, Int.(unwrap(x))], col)
            new_multi_col = [col for col in eachrow(hcat(new_multi_col...))]
            push!(new_cols, new_multi_col...)
            feat_names_with_inds = generate_new_feat_names(
                feat_name,
                size(level2vector, 1),
                feat_names,
            )
            push!(new_feat_names, feat_names_with_inds...)
        else
            # Not to be transformed => left as is
            push!(new_feat_names, feat_name)
            push!(new_cols, col)
        end
    end

    transformed_X = NamedTuple{tuple(new_feat_names...)}(tuple(new_cols)...)
    # Attempt to preserve table type
    transformed_X = Tables.materializer(X)(transformed_X)
    return transformed_X
end
