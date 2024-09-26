# # NEURAL NETWORK CLASSIFIER

seed!(1234)
N = 300
Xm = MLJBase.table(randn(Float32, N, 5));   # purely numeric
X = (; Tables.columntable(Xm)...,
    Column1 = repeat(Float32[1.0, 2.0, 3.0, 4.0, 5.0], Int(N / 5)),
    Column2 = categorical(repeat(['a', 'b', 'c', 'd', 'e'], Int(N / 5))),
    Column3 = categorical(repeat(["b", "c", "d", "f", "f"], Int(N / 5)), ordered = true),
    Column4 = repeat(Float32[1.0, 2.0, 3.0, 4.0, 5.0], Int(N / 5)),
    Column5 = randn(Float32, N),
    Column6 = categorical(
        repeat(["group1", "group1", "group2", "group2", "group3"], Int(N / 5)),
    ),
)

ycont = 2 * X.x1 - X.x3 + 0.1 * rand(Float32, N)
m, M = minimum(ycont), maximum(ycont)
_, a, b, _ = range(m, stop = M, length = 4) |> collect
y = map(ycont) do η
    if η < 0.9 * a
        'a'
    elseif η < 1.1 * b
        'b'
    else
        'c'
    end
end |> categorical;

# In the tests below we want to check GPU and CPU give similar results. We use the `MLP`
# builer instead of the default `Short()` because `Dropout()` in `Short()` does not appear
# to behave the same on GPU as on a CPU, even when we use `default_rng()` for both.

builder = MLJFlux.MLP(hidden = (8,))
optimiser = Optimisers.Adam(0.03)

losses = []

@testset_accelerated "NeuralNetworkClassifier" accel begin

    # Table input:
    @testset "Table input" begin
        basictest(MLJFlux.NeuralNetworkClassifier,
            X,
            y,
            builder,
            optimiser,
            0.85,
            accel)
    end

    @testset "Table input numerical" begin
        basictest(MLJFlux.NeuralNetworkClassifier,
            Xm,
            y,
            builder,
            optimiser,
            0.85,
            accel)
    end

    # Matrix input:
    @testset "Matrix input" begin
        basictest(MLJFlux.NeuralNetworkClassifier,
            matrix(Xm),
            y,
            builder,
            optimiser,
            0.85,
            accel)
    end

    train, test = MLJBase.partition(1:N, 0.7)

    # baseline loss (predict constant probability distribution):
    dict = StatsBase.countmap(y[train])
    prob_given_class = Dict{CategoricalArrays.CategoricalValue, Float64}()
    for (k, v) in dict
        prob_given_class[k] = dict[k] / length(train)
    end
    dist = MLJBase.UnivariateFinite(prob_given_class)
    loss_baseline =
        StatisticalMeasures.cross_entropy(fill(dist, length(test)), y[test])

    # check flux model is an improvement on predicting constant
    # distribution
    # (GPUs only support `default_rng`):
    rng = Random.default_rng()
    seed!(rng, 123)
    model = MLJFlux.NeuralNetworkClassifier(epochs = 50,
        builder = builder,
        optimiser = optimiser,
        acceleration = accel,
        batch_size = 10,
        rng = rng)
    @time mach = fit!(machine(model, X, y), rows = train, verbosity = 0)
    first_last_training_loss = MLJBase.report(mach)[1][[1, end]]
    push!(losses, first_last_training_loss[2])
    yhat = MLJBase.predict(mach, rows = test)
    @test StatisticalMeasures.cross_entropy(yhat, y[test]) < 0.95 * loss_baseline

    optimisertest(MLJFlux.NeuralNetworkClassifier,
        X,
        y,
        builder,
        optimiser,
        accel)

end

# check different resources (CPU1, CUDALibs, etc)) give about the same loss:
reference = losses[1]
println("losses for each resource: $losses")
@test all(x -> abs(x - reference) / reference < 5e-3, losses[2:end])


# # NEURAL NETWORK BINARY CLASSIFIER

@testset "NeuralNetworkBinaryClassifier constructor" begin
    model = MLJFlux.NeuralNetworkBinaryClassifier()
    @test model.loss == Flux.binarycrossentropy
    @test model.builder isa MLJFlux.Short
    @test model.finaliser == Flux.σ
end

seed!(1234)
N = 300
X = MLJBase.table(rand(Float32, N, 4));
ycont = Float32.(2 * X.x1 - X.x3 + 0.1 * rand(N))
m, M = minimum(ycont), maximum(ycont)
_, a, _ = range(m, stop = M, length = 3) |> collect
y = map(ycont) do η
    if η < 0.9 * a
        'a'
    else
        'b'
    end
end |> categorical;

builder = MLJFlux.MLP(hidden = (8,))
optimiser = Optimisers.Adam(0.03)

@testset_accelerated "NeuralNetworkBinaryClassifier" accel begin

    # Table input:
    @testset "Table input" begin
        basictest(
            MLJFlux.NeuralNetworkBinaryClassifier,
            X,
            y,
            builder,
            optimiser,
            0.85,
            accel,
        )
    end

    # Matrix input:
    @testset "Matrix input" begin
        basictest(
            MLJFlux.NeuralNetworkBinaryClassifier,
            matrix(X),
            y,
            builder,
            optimiser,
            0.85,
            accel,
        )
    end

    train, test = MLJBase.partition(1:N, 0.7)

    # baseline loss (predict constant probability distribution):
    dict = StatsBase.countmap(y[train])
    prob_given_class = Dict{CategoricalArrays.CategoricalValue, Float64}()
    for (k, v) in dict
        prob_given_class[k] = dict[k] / length(train)
    end
    dist = MLJBase.UnivariateFinite(prob_given_class)
    loss_baseline =
        StatisticalMeasures.cross_entropy(fill(dist, length(test)), y[test])

    # check flux model is an improvement on predicting constant
    # distribution
    # (GPUs only support `default_rng`):
    rng = Random.default_rng()
    seed!(rng, 123)
    model = MLJFlux.NeuralNetworkBinaryClassifier(
        epochs = 50,
        builder = builder,
        optimiser = optimiser,
        acceleration = accel,
        batch_size = 10,
        rng = rng,
    )
    @time mach = fit!(machine(model, X, y), rows = train, verbosity = 0)
    first_last_training_loss = MLJBase.report(mach)[1][[1, end]]
    yhat = MLJBase.predict(mach, rows = test)
    @test StatisticalMeasures.cross_entropy(yhat, y[test]) < 0.95 * loss_baseline

end

true
