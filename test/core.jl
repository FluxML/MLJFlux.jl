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
    Xmatrix = rand(10, 3)
    # convert to a column table:
    X = MLJBase.table(Xmatrix)

    y = rand(10)
    model = MLJFlux.NeuralNetworkRegressor()
    batch_size= 3
    @test MLJFlux.collate(model, X, y, batch_size) ==
        [(Xmatrix'[:,1:3], y[1:3]),
         (Xmatrix'[:,4:6], y[4:6]),
         (Xmatrix'[:,7:9], y[7:9]),
         (Xmatrix'[:,10:10], y[10:10])]

    # MultitargetNeuralNetworRegressor:
    ymatrix = rand(10, 2)
    y = MLJBase.table(ymatrix) # a rowaccess table
    model = MLJFlux.NeuralNetworkRegressor()
    batch_size= 3
    @test MLJFlux.collate(model, X, y, batch_size) ==
        [(Xmatrix'[:,1:3], ymatrix'[:,1:3]),
         (Xmatrix'[:,4:6], ymatrix'[:,4:6]),
         (Xmatrix'[:,7:9], ymatrix'[:,7:9]),
         (Xmatrix'[:,10:10], ymatrix'[:,10:10])]
    y = Tables.columntable(y) # try a columnaccess table
    @test MLJFlux.collate(model, X, y, batch_size) ==
        [(Xmatrix'[:,1:3], ymatrix'[:,1:3]),
         (Xmatrix'[:,4:6], ymatrix'[:,4:6]),
         (Xmatrix'[:,7:9], ymatrix'[:,7:9]),
         (Xmatrix'[:,10:10], ymatrix'[:,10:10])]
end

true
