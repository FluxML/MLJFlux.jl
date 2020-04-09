@testset "optimiser equality" begin
    @test Flux.Momentum() == Flux.Momentum()
    @test Flux.Momentum(0.1) != Flux.Momentum(0.2)
    @test Flux.ADAM(0.1) != Flux.ADAM(0.2)
end

@testset "nrows" begin
    Xmatrix = rand(10, 3)
    X = MLJBase.table(Xmatrix)
    @test MLJFlux.nrows(X) == 10
    @test MLJFlux.nrows(Tables.columntable(X)) == 10
end

@testset "collate" begin
    # NeuralNetworRegressor:
    Xmatrix = broadcast(x->round(x, sigdigits=2), rand(10, 3))
    # convert to a column table:
    X = MLJBase.table(Xmatrix)

    y = rand(10)
    model = MLJFlux.NeuralNetworkRegressor()
    model.batch_size= 3
    @test MLJFlux.collate(model, X, y) ==
        [(Xmatrix'[:,1:3], y[1:3]),
         (Xmatrix'[:,4:6], y[4:6]),
         (Xmatrix'[:,7:9], y[7:9]),
         (Xmatrix'[:,10:10], y[10:10])]

    # NeuralNetworClassifier:
    y = categorical([:a, :b, :a, :a, :b, :a, :a, :a, :b, :a])
    model = MLJFlux.NeuralNetworkClassifier()
    model.batch_size= 3
    data = MLJFlux.collate(model, X, y)
    @test first.(data) ==
        [Xmatrix'[:,1:3], Xmatrix'[:,4:6],
          Xmatrix'[:,7:9], Xmatrix'[:,10:10]]
    @test last.(data) ==
        [[1 0 1; 0 1 0], [1 0 1; 0 1 0],
         [1 1 0; 0 0 1], reshape([1; 0], (2,1))]

    # MultitargetNeuralNetworRegressor:
    ymatrix = rand(10, 2)
    y = MLJBase.table(ymatrix) # a rowaccess table
    model = MLJFlux.NeuralNetworkRegressor()
    model.batch_size= 3
    @test MLJFlux.collate(model, X, y) ==
        [(Xmatrix'[:,1:3], ymatrix'[:,1:3]),
         (Xmatrix'[:,4:6], ymatrix'[:,4:6]),
         (Xmatrix'[:,7:9], ymatrix'[:,7:9]),
         (Xmatrix'[:,10:10], ymatrix'[:,10:10])]
    y = Tables.columntable(y) # try a columnaccess table
    @test MLJFlux.collate(model, X, y) ==
        [(Xmatrix'[:,1:3], ymatrix'[:,1:3]),
         (Xmatrix'[:,4:6], ymatrix'[:,4:6]),
         (Xmatrix'[:,7:9], ymatrix'[:,7:9]),
         (Xmatrix'[:,10:10], ymatrix'[:,10:10])]

    # ImageClassifier
    Xmatrix = [rand(Float32, 6,6,1) for i=1:10]
    y = categorical([:a, :b, :a, :a, :b, :a, :a, :a, :b, :a])
    model = MLJFlux.ImageClassifier(batch_size=2)

    data = MLJFlux.collate(model, Xmatrix, y)
    @test  first.(data) ==
        [cat(Xmatrix[1], Xmatrix[2], dims=4),
        cat(Xmatrix[3], Xmatrix[4], dims=4),
        cat(Xmatrix[5], Xmatrix[6], dims=4),
        cat(Xmatrix[7], Xmatrix[8], dims=4),
        cat(Xmatrix[9], Xmatrix[10], dims=4),
        ]

    expected_y = [[1 0;0 1], [1 1;0 0], [0 1; 1 0], [1 1;0 0], [0 1; 1 0]]
    for i=1:5
        @test Int.(last.(data)[i]) == expected_y[i]
    end

end

@testset "fit!" begin

    Xmatrix = rand(100, 5)
    X = MLJBase.table(Xmatrix)
    y = Xmatrix[:, 1] + Xmatrix[:, 2] + Xmatrix[:, 3] +
        Xmatrix[:, 4] + Xmatrix[:, 5]

    data = [(Xmatrix'[:,1:20], y[1:20]),
            (Xmatrix'[:,21:40], y[21:40]),
            (Xmatrix'[:,41:60], y[41:60]),
            (Xmatrix'[:,61:80], y[61:80]),
            (Xmatrix'[:, 81:100], y[81:100])]

    initial_chain = Flux.Chain(Flux.Dense(5, 15),
                               Flux.Dropout(0.2),
                               Flux.Dense(15, 8),
                               Flux.Dense(8, 1))
    test_input = rand(5, 1)

    chain, history = MLJFlux.fit!(initial_chain,
                                  Flux.Optimise.ADAM(0.001),
                                  Flux.mse, 10, 0, 0, 3, data)

    @test length(history) == 10

    # Dropout should be inactive during test mode
    @test chain(test_input) == chain(test_input)

    # Loss should decrease at every epoch
    @test history == sort(history, rev=true)

end


true
