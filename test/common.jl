ModelType = MLJFlux.NeuralNetworkRegressor

@testset "clean!" begin
    model = @test_logs (:warn, r"`lambda") begin
        ModelType(lambda = -1)
    end
    @test model.lambda == 0

    model = @test_logs (:warn, r"`alpha") begin
        ModelType(alpha = -1)
    end
    @test model.alpha == 0

    model = @test_logs (:warn, r"`epochs") begin
        ModelType(epochs = -1)
    end
    @test model.epochs == 10
    
    model = @test_logs (:warn, r"`batch_size") begin
        ModelType(batch_size = -1)
    end
    @test model.batch_size == 1
    
    if MLJFlux.gpu_isdead()
        model  = @test_logs (:warn, r"`acceleration") begin
            ModelType(acceleration=CUDALibs())
        end
        @test model.acceleration == CUDALibs()
    end
end
