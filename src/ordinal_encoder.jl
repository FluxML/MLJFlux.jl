"""
**Private Method**

Returns the indices of the categorical columns in the table `X`.
"""
function get_cat_inds(X)
    types = schema(X).scitypes
    cat_inds = findall(x -> x <: Finite, types)    
    return cat_inds
end


"""
**Private Method**

Fits an ordinal encoder to the table `X`, using only the columns with indices in `featinds`.

Returns a dictionary mapping each column index to a dictionary mapping each level in that column to an integer.
"""
function ordinal_encoder_fit(X; featinds)
    # 1. Define mapping per column per level dictionary
    mapping_per_feat_level = Dict()

    # 2. Use feature mapper to compute the mapping of each level in each column
    for i in featinds
        feat_col = Tables.getcolumn(X, i)
        feat_levels = levels(feat_col)
        # Compute the dict using the given feature_mapper function
        mapping_per_feat_level[i] = Dict{Any, Integer}(value => index for (index, value) in enumerate(feat_levels))
    end
    return mapping_per_feat_level
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
        error("While transforming, found novel levels for the column: $(lost_levels) that were not seen while training.")
    end
end

"""
**Private Method**

Transforms the table `X` using the ordinal encoder defined by `mapping_per_feat_level`.

Returns a new table with the same column names as `X`, but with categorical columns replaced by integer columns.
"""
function ordinal_encoder_transform(X, mapping_per_feat_level)
    feat_names = Tables.schema(X).names
    numfeats = length(feat_names)
    new_feats = []
    for ind in 1:numfeats
        col = Tables.getcolumn(X, ind)

        # Create the transformation function for each column
        if ind in keys(mapping_per_feat_level)
            train_levels = keys(mapping_per_feat_level[ind])
            test_levels = levels(col)
            check_unkown_levels(train_levels, test_levels)
            level2scalar = mapping_per_feat_level[ind]
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
