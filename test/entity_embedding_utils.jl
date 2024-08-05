
@testset "Embedding Enabled Types" begin
    clf = MLJFlux.NeuralNetworkClassifier(
        builder = MLJFlux.Short(n_hidden = 5, dropout = 0.2),
        optimiser = Optimisers.Adam(0.01),
        batch_size = 8,
        epochs = 100,
    )
    @test MLJFlux.is_embedding_enabled_type(typeof(clf))

    clf = MLJFlux.ImageClassifier(
        batch_size = 50,
        epochs = 10,
        rng = 123,
    )
    @test !MLJFlux.is_embedding_enabled_type(typeof(clf))
end


@testset "set_default_new_embedding_dim" begin
    # <= 20
    @test MLJFlux.set_default_new_embedding_dim(10) == 5
    @test MLJFlux.set_default_new_embedding_dim(15) == 8
    # > 20
    @test MLJFlux.set_default_new_embedding_dim(25) == 5
    @test MLJFlux.set_default_new_embedding_dim(30) == 6
end

@testset "check_mismatch_in_cat_feats" begin
    # Test with no mismatch
    featnames = [:a, :b, :c]
    cat_inds = [1, 3]
    specified_featinds = [1, 3]
    @test !MLJFlux.check_mismatch_in_cat_feats(featnames, cat_inds, specified_featinds)

    # Test with mismatch
    featnames = [:a, :b, :c]
    cat_inds = [1, 3]
    specified_featinds = [1, 2, 3]
    @test_throws ArgumentError MLJFlux.check_mismatch_in_cat_feats(featnames, cat_inds, specified_featinds)

    # Test with empty specified_featinds
    featnames = [:a, :b, :c]
    cat_inds = [1, 3]
    specified_featinds = []
    @test !MLJFlux.check_mismatch_in_cat_feats(featnames, cat_inds, specified_featinds)

    # Test with empty cat_inds
    featnames = [:a, :b, :c]
    cat_inds = []
    specified_featinds = [1, 2]
    @test_throws ArgumentError MLJFlux.check_mismatch_in_cat_feats(featnames, cat_inds, specified_featinds)
end

@testset "Testing set_new_embedding_dims" begin
    # Test case 1: Correct calculation of embedding dimensions when specified as floats
    featnames = ["color", "size", "type"]
    cat_inds = [1, 2]
    num_levels = [3, 5]
    embedding_dims = Dict("color" => 0.5, "size" => 2)
    
    result = MLJFlux.set_new_embedding_dims(featnames, cat_inds, num_levels, embedding_dims)
    @test result == [2, 2]  # Expected to be ceil(1.5) = 2 for "color", and exact 2 for "size"

    # Test case 2: Handling of unspecified dimensions with defaults
    embedding_dims = Dict("color" => 0.5)  # "size" is not specified
    result = MLJFlux.set_new_embedding_dims(featnames, cat_inds, num_levels, embedding_dims)
    @test result == [2, MLJFlux.set_default_new_embedding_dim(5)]  # Expected to be ceil(1.5) = 2 for "color", and default 1 for "size"

    # Test case 3: All embedding dimensions are unspecified, default for all
    embedding_dims = Dict()
    result = MLJFlux.set_new_embedding_dims(featnames, cat_inds, num_levels, embedding_dims)
    @test result == [MLJFlux.set_default_new_embedding_dim(3), MLJFlux.set_default_new_embedding_dim(5)]  # Default dimensions for both
end

@testset "test get_cat_inds" begin
    X = (
        C1 = [1, 2, 3, 4, 5],
        C2 = ['a', 'b', 'c', 'd', 'e'],
        C3 = ["b", "c", "d", "e", "f"],
        C4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    X = coerce(X, :C1 => OrderedFactor, :C2 => Multiclass, :C3 => Multiclass)
    @test MLJFlux.get_cat_inds(X) == [1, 2, 3]
end

@testset "Number of levels" begin
    X = (
        C1 = [1, 2, 3, 4, 5],
        C2 = ['a', 'b', 'c', 'd', 'e'],
        C3 = ["b", "c", "d", "f", "f"],
        C4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    @test MLJFlux.get_num_levels(X, [2]) == [5]
    @test MLJFlux.get_num_levels(X, [2, 3]) == [5, 4]
end


@testset "Testing prepare_entityembs" begin
    X = (
        Column1 = [1, 2, 3, 4, 5],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = categorical(["b", "c", "d"]),
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )

    featnames = [:Column1, :Column2, :Column3, :Column4]
    cat_inds = [2, 3]  # Assuming categorical columns are 2 and 3
    embedding_dims = Dict(:Column2 => 3, :Column3 => 2)

    entityprops_expected = [
        (index = 2, levels = 5, newdim = 3),
        (index = 3, levels = 3, newdim = 2),
    ]
    output_dim_expected = 3 + 2 + 4 - 2  # Total embedding dims + non-categorical features

    entityprops, entityemb_output_dim =
        MLJFlux.prepare_entityembs(X, featnames, cat_inds, embedding_dims)

    @test entityprops == entityprops_expected
    @test entityemb_output_dim == output_dim_expected
end

