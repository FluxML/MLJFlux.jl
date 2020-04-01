# Multitarget NN Regressor

Random.seed!(123)
N = 20
X = MLJBase.table(randn(Float32, 10N, 5));

@testset "Regressors" begin

    exs =[:(MLJFlux.MultitargetNeuralNetworkRegressor),
          :(MLJFlux.NeuralNetworkRegressor)]

    for ModelType in exs

        if eval(ModelType) <: MLJFlux.Regressor
            builder = MLJFlux.Short(σ=identity)
            if eval(ModelType) <: MLJFlux.MultitargetNeuralNetworkRegressor
                ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5);
                y = Tables.table(ymatrix);
                truth =  ymatrix[test, :]
            else
                y = 1 .+ X.x1 - X.x2 .- 2X.x4 + X.x5
                truth =  y[test]
            end
        end

        eval(quote

             @info "Testing $($ModelType). "

             train = 1:7N
             test = (7N+1):10N

             model = $ModelType(builder=$builder)

             fitresult, cache, report =
             MLJBase.fit(model,
                         0,
                         MLJBase.selectrows(X, train),
                         MLJBase.selectrows($y, train));

             history = report.training_losses;
             @test length(history) == model.epochs
             best = history[end]
             @test best < 0.6*history[1]

             # Update model without retraining and check no restart:
             model.epochs = 3
             fitresult, cache, report =
             @test_logs((:info, r""), # one line of :info per extra epoch
                        (:info, r""),
                        (:info, r""),
                        MLJBase.update(model,
                                       2,
                                       fitresult,
                                       cache,
                                       MLJBase.selectrows(X,train),
                                       MLJBase.selectrows($y, train)));

             yhat = MLJBase.predict(model,
                                    fitresult,
                                    MLJBase.selectrows(X, test))

             # is at least a bit better than constant predictor:
             if $ModelType <: MLJFlux.Regressor
                 goal =0.9*model.loss(truth .- mean(truth), 0)
                 if $ModelType <: MLJFlux.MultitargetNeuralNetworkRegressor
                     @test model.loss(Tables.matrix(yhat), $truth) < goal
                 else
                     @test model.loss(yhat, $truth) < goal
                 end
             end

             history = report.training_losses;
             @test length(history) == model.epochs

             # start fresh with small epochs:
             model = $ModelType(builder=$builder, epochs=2)
             fitresult, cache, report =
             MLJBase.fit(model,
                         0,
                         MLJBase.selectrows(X, train),
                         MLJBase.selectrows($y, train));

             # change batch_size and check it restarts:
             model.batch_size = 2
             fitresult, cache, report =
             @test_logs((:info, r""), # one line of :info per extra epoch
                        (:info, r""),
                        MLJBase.update(model,
                                       2,
                                       fitresult,
                                       cache,
                                       MLJBase.selectrows(X,train),
                                       MLJBase.selectrows($y, train)));

             # change learning rate and check it does *not* restart:
             model.optimiser.eta /= 2
             fitresult, cache, report =
             @test_logs(MLJBase.update(model,
                                       2,
                                       fitresult,
                                       cache,
                                       MLJBase.selectrows(X,train),
                                       MLJBase.selectrows($y, train)));

             # set `optimiser_changes_trigger_retraining = true` and change
             # learning rate and check it does restart:
             model.optimiser_changes_trigger_retraining = true
             model.optimiser.eta /= 2
             @test_logs((:info, r""), # one line of :info per extra epoch
                        (:info, r""),
                        MLJBase.update(model,
                                       2,
                                       fitresult,
                                       cache,
                                       MLJBase.selectrows(X,train),
                                       MLJBase.selectrows($y, train)));
             end)
    end
end

true
