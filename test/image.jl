# # BASIC IMAGE TESTS GREY

Random.seed!(123)
stable_rng = StableRNGs.StableRNG(123)

mutable struct MyNeuralNetwork <: MLJFlux.Builder
    kernel1
    kernel2
end

function MLJFlux.build(model::MyNeuralNetwork, rng, ip, op, n_channels)
    init = Flux.glorot_uniform(rng)
    Flux.Chain(
        Flux.Conv(model.kernel1, n_channels=>2, init=init),
        Flux.Conv(model.kernel2, 2=>1, init=init),
        x->reshape(x, :, size(x)[end]),
        Flux.Dense(16, op, init=init))
end

builder = MyNeuralNetwork((2,2), (2,2))
images, labels = MLJFlux.make_images(stable_rng)
losses = []

@testset_accelerated "ImageClassifier basic tests" accel begin

    Random.seed!(123)
    stable_rng = StableRNGs.StableRNG(123)

    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    acceleration=accel,
                                    rng=stable_rng)

    fitresult, cache, _report = MLJBase.fit(model, 0, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    model.epochs = 15
    MLJBase.update(model, 0, fitresult, cache, images, labels)

    pred = MLJBase.predict(model, fitresult, images[1:6])

    # try with batch_size > 1:
    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    batch_size=2,
                                    acceleration=accel,
                                    rng=stable_rng)
    model.optimiser.eta = 0.005
    @time fitresult, cache, _report = MLJBase.fit(model, 0, images, labels);
    first_last_training_loss = _report[1][[1, end]]
    push!(losses, first_last_training_loss[2])
#    @show first_last_training_loss

    # tests update logic, etc (see test_utililites.jl):
    @test basictest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, 0.95, accel)

    @test optimisertest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, accel)

end

# check different resources (CPU1, CUDALibs) give about the same loss:
reference = losses[1]
@info "Losses for each computational resource: $losses"
@test all(x->abs(x - reference)/reference < 5e-4, losses[2:end])


# # BASIC IMAGE TESTS COLOR

builder = MyNeuralNetwork((2,2), (2,2))
images, labels = MLJFlux.make_images(stable_rng, color=true)
losses = []

@testset_accelerated "ColorImages" accel begin

    Random.seed!(123)
    stable_rng = StableRNGs.StableRNG(123)

    model = MLJFlux.ImageClassifier(builder=builder,
                                    epochs=10,
                                    acceleration=accel,
                                    rng=stable_rng)
    # tests update logic, etc (see test_utililites.jl):
    @test basictest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, 0.95, accel)

    @time fitresult, cache, _report = MLJBase.fit(model, 0, images, labels);
    pred = MLJBase.predict(model, fitresult, images[1:6])
    first_last_training_loss = _report[1][[1, end]]
    push!(losses, first_last_training_loss[2])

    # try with batch_size > 1:
    model = MLJFlux.ImageClassifier(epochs=10,
                                    builder=builder,
                                    batch_size=2,
                                    acceleration=accel,
                                    rng=stable_rng)
    fitresult, cache, _report = MLJBase.fit(model, 0, images, labels);

    @test optimisertest(MLJFlux.ImageClassifier, images, labels,
                           model.builder, model.optimiser, accel)

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
@info "Losses for each computational resource: $losses"
@test all(x->abs(x - reference)/reference < 1e-5, losses[2:end])


# # SMOKE TEST FOR DEFAULT BUILDER

images, labels = MLJFlux.make_images(stable_rng, image_size=(32, 32), n_images=12,
noise=0.2, color=true);

@testset_accelerated "ImageClassifier basic tests" accel begin
    model = MLJFlux.ImageClassifier(epochs=5,
                                    batch_size=4,
                                    acceleration=accel,
                                    rng=stable_rng)
    fitresult, _, _ = MLJBase.fit(model, 0, images, labels);
    predict(model, fitresult, images)
end

true
