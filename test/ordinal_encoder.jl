@testset "test get_cat_inds" begin
    X = (
        C1 = [1, 2, 3, 4, 5],
        C2 = ['a', 'b', 'c', 'd', 'e'],
        C3 = ["b", "c", "d", "e", "f"],
        C4 = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
    X = coerce(X, :C1=>OrderedFactor,:C2=>Multiclass, :C3=>Multiclass)
    @test MLJFlux.get_cat_inds(X) == [1, 2, 3]    
end


@testset "ordinal encoder" begin
    X = (
        Column1 = [1, 2, 3, 4, 5],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = categorical(["b", "c", "d"]),
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
    map = MLJFlux.ordinal_encoder_fit(X; featinds = [2, 3])
    Xenc = MLJFlux.ordinal_encoder_transform(X, map)
    @test map[2] == Dict('a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5)
    @test map[3] == Dict("b" => 1, "c" => 2, "d" => 3 )
    @test Xenc.Column1 == [1, 2, 3, 4, 5]
    @test Xenc.Column2 == [1, 2, 3, 4, 5]
    @test Xenc.Column3 == [1, 2, 3]
    @test Xenc.Column4 == [1.0, 2.0, 3.0, 4.0, 5.0]
end



