

@testset "ordinal encoder" begin
    X = (
        Column1 = [1.0, 2.0, 3.0, 4.0, 5.0],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = categorical(["b", "c", "d"]),
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    # Test Encoding Functionality
    map = MLJFlux.ordinal_encoder_fit(X; featinds = [2, 3])
    Xenc = MLJFlux.ordinal_encoder_transform(X, map)
    @test map[2] == Dict('a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5)
    @test map[3] == Dict("b" => 1, "c" => 2, "d" => 3)
    @test Xenc.Column1 == [1.0, 2.0, 3.0, 4.0, 5.0]
    @test Xenc.Column2 == Float32.([1.0, 2.0, 3.0, 4.0, 5.0])
    @test Xenc.Column3 == Float32.([1, 2, 3])
    @test Xenc.Column4 == [1.0, 2.0, 3.0, 4.0, 5.0]

    X = coerce(X, :Column1 => Multiclass)
    map = MLJFlux.ordinal_encoder_fit(X; featinds = [1, 2, 3])
    @test !haskey(map, 1)   # already encoded

    @test Xenc == MLJFlux.ordinal_encoder_fit_transform(X; featinds = [2, 3])[1]

    # Test Consistency with Types
    scs = schema(Xenc).scitypes
    ts  = schema(Xenc).types
    
    # 1) all scitypes must be exactly Continuous
    @test all(scs .== Continuous)
    
    # 2) all types must be a concrete subtype of AbstractFloat (i.e. <: AbstractFloat, but ≠ AbstractFloat itself)
    @test all(t -> t <: AbstractFloat && isconcretetype(t), ts)
end

@testset "Generate New feature names Function Tests" begin
    # Test 1: No initial conflicts
    @testset "No Initial Conflicts" begin
        existing_names = []
        names = MLJFlux.generate_new_feat_names("feat", 3, existing_names)
        @test names == [Symbol("feat_1"), Symbol("feat_2"), Symbol("feat_3")]
    end

    # Test 2: Handle initial conflict by adding underscores
    @testset "Initial Conflict Resolution" begin
        existing_names = [Symbol("feat_1"), Symbol("feat_2"), Symbol("feat_3")]
        names = MLJFlux.generate_new_feat_names("feat", 3, existing_names)
        @test names == [Symbol("feat__1"), Symbol("feat__2"), Symbol("feat__3")]
    end
end


@testset "embedding_transform works" begin
    X = (
        Column1 = [1.0, 2.0, 3.0, 4.0, 5.0],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = categorical(["b", "c", "d", "f", "f"]),
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    mapping_matrices = Dict(
        :Column2 => [
            1 0.5 0.7 4 5
            0.4 2 3 0.9 0.2
            0.1 0.6 0.8 0.3 0.4
        ],
        :Column3 => [
            1 0.5 0.7 4
            0.4 2 3 0.9
        ],
    )
    X, _ = MLJFlux.ordinal_encoder_fit_transform(X; featinds = [2, 3])
    Xenc = MLJFlux.embedding_transform(X, mapping_matrices)
    @test Xenc == (
        Column1 = [1.0, 2.0, 3.0, 4.0, 5.0],
        Column2_1 = [1.0, 0.5, 0.7, 4.0, 5.0],
        Column2_2 = [0.4, 2.0, 3.0, 0.9, 0.2],
        Column2_3 = [0.1, 0.6, 0.8, 0.3, 0.4],
        Column3_1 = [1.0, 0.5, 0.7, 4.0, 4.0],
        Column3_2 = [0.4, 2.0, 3.0, 0.9, 0.9],
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )
end
