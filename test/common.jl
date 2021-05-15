ModelType = MLJFlux.NeuralNetworkRegressor

@test "equality" begin
    model1 = ImageClassifier()
    @test model1 == deepcopy(ImageClassifier())
end

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

@testset "iteration api" begin
    model = MLJFlux.NeuralNetworkRegressor(epochs=10)
    @test MLJBase.supports_training_losses(model)
    @test MLJBase.iteration_parameter(model) == fieldnames(typeof(model))[4]

    # integration test:
    X, y = MLJBase.make_regression(10)
    mach = MLJBase.machine(model, X, y)
    MLJBase.fit!(mach, verbosity=0)
    losses = MLJBase.training_losses(mach)
    @test losses == MLJBase.report(mach).training_losses[2:end]
    @test length(losses) == 10
end

