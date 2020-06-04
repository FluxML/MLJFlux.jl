# To run a battery of tests checking: (i) fit, predict & update calls
# work; (ii) update logic is correct; (iii) training loss after 10
# epochs is 80% or better than initial loss:
function basictest(ModelType, X, y, builder, optimiser)

    ModelType_ex = Meta.parse(string(ModelType))

    eval(quote

         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser)

         fitresult, cache, report =
         MLJBase.fit(model,0, $X, $y);

         history = report.training_losses;
         @test length(history) == model.epochs

         # test improvement in training loss:
         @test history[end] < 0.8*history[1]

         # increase iterations and check update is incremental:
         model.epochs = model.epochs + 3

         fitresult, cache, report =
         @test_logs((:info, r""), # one line of :info per extra epoch
                    (:info, r""),
                    (:info, r""),
                    MLJBase.update(model, 2,fitresult, cache, $X, $y));

         yhat = MLJBase.predict(model, fitresult, $X)

         history = report.training_losses;
         @test length(history) == model.epochs

         # start fresh with small epochs:
         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser,
                               epochs=2)
         fitresult, cache, report = MLJBase.fit(model, 0, $X, $y);

         # change batch_size and check it performs cold restart:
         model.batch_size = 2
         fitresult, cache, report =
         @test_logs((:info, r""), # one line of :info per extra epoch
                    (:info, r""),
                    MLJBase.update(model, 2, fitresult, cache, $X, $y ));

         # change learning rate and check it does *not* restart:
         model.optimiser.eta /= 2
         fitresult, cache, report =
         @test_logs(MLJBase.update(model, 2, fitresult, cache, $X, $y));

         # set `optimiser_changes_trigger_retraining = true` and change
         # learning rate and check it does restart:
         model.optimiser_changes_trigger_retraining = true
         model.optimiser.eta /= 2
         @test_logs((:info, r""), # one line of :info per extra epoch
                    (:info, r""),
                    MLJBase.update(model, 2, fitresult, cache, $X, $y));

         end)

    return true
end
