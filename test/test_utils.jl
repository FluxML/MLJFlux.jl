macro testset_accelerated(name::String, var, ex)
    testset_accelerated(name, var, ex)
end
macro testset_accelerated(name::String, var, opts::Expr, ex)
    testset_accelerated(name, var, ex; eval(opts)...)
end

# To exclude a resource, say, CPU1, do like
# `@test_accelerated "cool test" accel (exclude=[CPU1,],) begin ... end`
function testset_accelerated(name::String, var, ex; exclude=[])

    @info "Starting this test: $name..."
    
    final_ex = quote end

    append!(exclude, EXCLUDED_RESOURCE_TYPES)

    for res in RESOURCES
        if any(x->typeof(res)<:x, exclude)
            push!(final_ex.args, quote
               $var = $res
               @testset $(name*" ($(typeof(res).name))") begin
                   @test_broken false
               end
            end)
        else
            push!(final_ex.args, quote
               $var = $res
               @testset $(name*" ($(typeof(res).name))") $ex
            end)
        end
    end
    return esc(final_ex)
end


# To run a battery of tests checking: (i) fit, predict & update calls
# work; (ii) update logic is correct; (iii) training loss after 10
# epochs is better than `threshold` times initial loss:
function basictest(ModelType, X, y, builder, optimiser, threshold, accel)

    ModelType_str = string(ModelType)
    ModelType_ex = Meta.parse(ModelType_str)
    accel_ex = Meta.parse(string(accel))

    eval(quote

         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser,
                               acceleration=$accel_ex)

         fitresult, cache, report =
         MLJBase.fit(model,0, $X, $y);

         history = report.training_losses;
         @test length(history) == model.epochs

         # test improvement in training loss:
         @test history[end] < $threshold*history[1]

         # increase iterations and check update is incremental:
         model.epochs = model.epochs + 3

         fitresult, cache, report =
         @test_logs((:info, r""), # one line of :info per extra epoch
                    (:info, r""),
                    (:info, r""),
                    MLJBase.update(model, 2,fitresult, cache, $X, $y));

         @test MLJBase.fitted_params(model, fitresult).chain isa Flux.Chain

         yhat = MLJBase.predict(model, fitresult, $X)

         history = report.training_losses;
         @test length(history) == model.epochs

         # start fresh with small epochs:
         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser,
                               epochs=2,
                               acceleration=$accel_ex)
         println()
         fitresult, cache, report = MLJBase.fit(model, 1, $X, $y);
         println()

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
