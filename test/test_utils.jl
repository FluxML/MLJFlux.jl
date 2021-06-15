seed!(::CPU1, i=123) = Random.seed!(i)
seed!(::CUDALibs, i=123) = Flux.CUDA.seed!(i)

macro testset_accelerated(name::String, var, ex)
    testset_accelerated(name, var, ex)
end
macro testset_accelerated(name::String, var, opts::Expr, ex)
    testset_accelerated(name, var, ex; eval(opts)...)
end

# To exclude a resource, say, CPU1, do like
# `@test_accelerated "cool test" accel (exclude=[CPU1,],) begin ... end`
function testset_accelerated(name::String, var, ex; exclude=[])

    final_ex = quote end

    push!(final_ex.args, quote
          println()
          @info "$($name):"
          end)

    append!(exclude, EXCLUDED_RESOURCE_TYPES)

    for res in RESOURCES
        push!(final_ex.args, quote
              @info "acceleration = $($res)"
              end)
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
    optimiser = deepcopy(optimiser)

    eval(quote

         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser,
                               acceleration=$accel_ex)

         fitresult, cache, _report = MLJBase.fit(model, 0, $X, $y);

         history = _report.training_losses;
         @test length(history) == model.epochs + 1

         # test improvement in training loss:
         @test history[end] < $threshold*history[1]

         # increase iterations and check update is incremental:
         model.epochs = model.epochs + 3

         fitresult, cache, _report =
         @test_logs((:info, r""), # one line of :info per extra epoch
                    (:info, r""),
                    (:info, r""),
                    MLJBase.update(model, 2,fitresult, cache, $X, $y));

         @test :chain in keys(MLJBase.fitted_params(model, fitresult))

         yhat = MLJBase.predict(model, fitresult, $X)

         history = _report.training_losses;
         @test length(history) == model.epochs + 1

         # start fresh with small epochs:
         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser,
                               epochs=2,
                               acceleration=$accel_ex)

         fitresult, cache, _report = MLJBase.fit(model, 0, $X, $y);

         # change batch_size and check it performs cold restart:
         model.batch_size = 2
         fitresult, cache, _report =
         @test_logs((:info, r""), # one line of :info per extra epoch
                    (:info, r""),
                    MLJBase.update(model, 2, fitresult, cache, $X, $y ));

         # change learning rate and check it does *not* restart:
         model.optimiser.eta /= 2
         fitresult, cache, _report =
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

# to test the optimiser "state" is preserved in update (warm restart):
function optimisertest(ModelType, X, y, builder, optimiser, accel)

    ModelType_str = string(ModelType)
    ModelType_ex = Meta.parse(ModelType_str)
    accel_ex = Meta.parse(string(accel))
    optimiser = deepcopy(optimiser)

    eval(quote
         model = $ModelType_ex(builder=$builder,
                               optimiser=$optimiser,
                               acceleration=$accel_ex,
                               epochs=1)

             mach = machine(model, $X, $y);

             # two epochs in stages:
             Random.seed!(123) # chains are always initialized on CPU
             fit!(mach, verbosity=0, force=true);
             model.epochs = model.epochs + 1
             fit!(mach, verbosity=0); # update
             l1 = MLJBase.report(mach).training_losses[end]

             # two epochs in one go:
             Random.seed!(123) # chains are always initialized on CPU
             fit!(mach, verbosity=1, force=true)
             l2 = MLJBase.report(mach).training_losses[end]

             if accel isa CPU1
                 @test isapprox(l1, l2)
             else
                 @test_broken isapprox(l1, l2, rtol=1e-8)
             end

         end)

    return true
end
