"""
See more functional tests in entity_embedding_utils.jl and mlj_model_interface.jl
"""
batch = Float32.([
    0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1;
    1   2   3   4   5   6   7   8   9  10;
    0.9 0.1 0.4 0.5 0.3 0.7 0.8 0.9 1.0 1.1;
    1   1   2   2   1   1   2   2   1   1
])


entityprops = [
    (index = 2, levels = 10, newdim = 2),
    (index = 4, levels = 2, newdim = 1),
]


@testset "Feedforward with Entity Embedder Works" begin
    ### Option 1: Use EntityEmbedder
    entityprops = [
        (index = 2, levels = 10, newdim = 5),
        (index = 4, levels = 2, newdim = 2),
    ]

    embedder = MLJFlux.EntityEmbedderLayer(entityprops, 4)

    output = embedder(batch)

    ### Option 2: Manual feedforward
    x1 = batch[1:1, :]
    z2 = Int.(batch[2, :])
    x3 = batch[3:3, :]
    z4 = Int.(batch[4, :])

    # extract matrices from categorical embedder
    EE1 = Flux.params(embedder.embedders[2])[1]         # (newdim, levels) = (5, 10)
    EE2 = Flux.params(embedder.embedders[4])[1]         # (newdim, levels) = (2, 2)

    ## One-hot encoding 
    z2_hot = Flux.onehotbatch(z2, levels(z2))
    z4_hot = Flux.onehotbatch(z4, levels(z4))

    function feedforward(x1, z2_hot, x3, z4_hot)
        f_z2 = EE1 * z2_hot
        f_z4 = EE2 * z4_hot
        return vcat([x1, f_z2, x3, f_z4]...)
    end

    real_output = feedforward(x1, z2_hot, x3, z4_hot)
    @test output ≈ real_output
end


@testset "Feedforward and Backward Pass with Entity Embedder Works" begin
    y_batch_reg = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0] # Regression
    y_batch_cls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]           # Classification
    y_batch_cls_o = Flux.onehotbatch(y_batch_cls, 1:10)

    losses = [Flux.crossentropy, Flux.mse]
    targets = [y_batch_cls_o, y_batch_reg]
    finalizer = [softmax, relu]

    for ind in 1:2
        ### Option 1: Feedforward with EntityEmbedder in the network
        entityprops = [
            (index = 2, levels = 10, newdim = 5),
            (index = 4, levels = 2, newdim = 2),
        ]

        cat_model = Chain(
            MLJFlux.EntityEmbedderLayer(entityprops, 4),
            Dense(9 => (ind == 1) ? 10 : 1),
            finalizer[ind],
        )

        EE1_before = Flux.params(cat_model.layers[1].embedders[2])[1]
        EE2_before = Flux.params(cat_model.layers[1].embedders[4])[1]
        W_before = Flux.params(cat_model.layers[2])[1]

        ### Test with obvious equivalent feedforward
        x1 = batch[1:1, :]
        z2 = Int.(batch[2, :])
        x3 = batch[3:3, :]
        z4 = Int.(batch[4, :])

        z2_hot = Flux.onehotbatch(z2, levels(z2))
        z4_hot = Flux.onehotbatch(z4, levels(z4))

        ### Option 2: Manual feedforward
        function feedforward(x1, z2_hot, x3, z4_hot, W, EE1, EE2)
            f_z2 = EE1 * z2_hot
            f_z4 = EE2 * z4_hot
            return finalizer[ind](W * vcat([x1, f_z2, x3, f_z4]...))
        end

        struct ObviousNetwork
            W::Any
            EE1::Any
            EE2::Any
        end

        (m::ObviousNetwork)(x1, z2_hot, x3, z4_hot) =
            feedforward(x1, z2_hot, x3, z4_hot, m.W, m.EE1, m.EE2)
        Flux.@layer ObviousNetwork

        W_before_cp, EE1_before_cp, EE2_before_cp =
            deepcopy(W_before), deepcopy(EE1_before), deepcopy(EE2_before)
        net = ObviousNetwork(W_before_cp, EE1_before_cp, EE2_before_cp)

        @test feedforward(x1, z2_hot, x3, z4_hot, W_before, EE1_before, EE2_before) ≈
              cat_model(batch)

        ## Option 1: Backward with EntityEmbedder in the network
        loss, grads = Flux.withgradient(cat_model) do m
            y_pred_cls = m(batch)
            losses[ind](y_pred_cls, targets[ind])
        end
        optim = Flux.setup(Flux.Adam(10), cat_model)
        new_params = Flux.update!(optim, cat_model, grads[1])

        EE1_after = Flux.params(new_params[1].layers[1].embedders[2].weight)[1]
        EE2_after = Flux.params(new_params[1].layers[1].embedders[4].weight)[1]
        W_after = Flux.params(new_params[1].layers[2].weight)[1]

        ## Option 2: Backward with ObviousNetwork
        loss, grads = Flux.withgradient(net) do m
            y_pred_cls = m(x1, z2_hot, x3, z4_hot)
            losses[ind](y_pred_cls, targets[ind])
        end

        optim = Flux.setup(Flux.Adam(10), net)
        z = Flux.update!(optim, net, grads[1])
        EE1_after_cp = Flux.params(z[1].EE1)[1]
        EE2_after_cp = Flux.params(z[1].EE2)[1]
        W_after_cp = Flux.params(z[1].W)[1]
        @test EE1_after_cp ≈ EE1_after
        @test EE2_after_cp ≈ EE2_after
        @test W_after_cp ≈ W_after
    end
end


@testset "Transparent when no categorical variables" begin
    entityprops = []
    numfeats = 4
    embedder = MLJFlux.EntityEmbedderLayer(entityprops, 4)
    output = embedder(batch)
    @test output ≈ batch
    @test eltype(output) == Float32
end


@testset "get_embedding_matrices works and has the right dimensions" begin
    models = [
        MLJFlux.NeuralNetworkBinaryClassifier,
        MLJFlux.NeuralNetworkClassifier,
        MLJFlux.NeuralNetworkRegressor,
        MLJFlux.MultitargetNeuralNetworkRegressor,
    ]

    X = (
        Column1 = [1.0, 2.0, 3.0, 4.0, 5.0],
        Column2 = categorical(['a', 'b', 'c', 'd', 'e']),
        Column3 = categorical(["b", "c", "d", "f", "f"], ordered = true),
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0],
        Column5 = randn(5),
        Column6 = categorical(["group1", "group1", "group2", "group2", "group3"]),
    )

    y = categorical([0, 1, 0, 1, 1])
    yreg = [0.1, -0.3, 0.2, 0.8, 0.9]
    ys = [y, y, yreg, yreg]

    embedding_dims = [
        Dict(:Column2 => 0.5, :Column3 => 2, :Column6 => 0.1),
        Dict(:Column2 => 1, :Column3 => 4),
        Dict(),
    ]
    expected_dims = [
        [(3, 5), (2, 4), (1, 3)],
        [(1, 5), (4, 4), (2, 3)],
        [(4, 5), (3, 4), (2, 3)],
    ]

    size([
        1 2
        3 4
    ])

    stable_rng=StableRNG(123)

    for j in eachindex(embedding_dims)
        for i in eachindex(models)
            # Without lightweight wrapper
            clf = models[1](
                builder = MLJFlux.Short(n_hidden = 5, dropout = 0.0),
                optimiser = Optimisers.Adam(0.01),
                batch_size = 8,
                epochs = 100,
                acceleration = CUDALibs(),
                optimiser_changes_trigger_retraining = true,
                embedding_dims = embedding_dims[3],
                rng=42
            )
            mach = machine(clf, X, ys[1])
            fit!(mach, verbosity = 0)
            Xnew = transform(mach, X)
            # With lightweight wrapper
            clf2 = deepcopy(clf)
            emb = MLJFlux.EntityEmbedder(clf2)
            mach_emb = machine(emb, X, ys[1])
            fit!(mach_emb, verbosity = 0)
            Xnew_emb = transform(mach_emb, X)
            @test Xnew == Xnew_emb

            # Pipeline doesn't throw an error
            pipeline = emb |> clf
            mach_pipe = machine(pipeline, X, y)
            fit!(mach_pipe, verbosity = 0)
            y = predict_mode(mach_pipe, X)

            mapping_matrices = MLJFlux.get_embedding_matrices(
                fitted_params(mach).chain,
                [2, 3, 6],
                [:Column1, :Column2, :Column3, :Column4, :Column5, :Column6],
            )

            embedder_layer = fitted_params(mach).chain.layers[1]
            # get_embedding_matrices work
            @test mapping_matrices[:Column2] == Flux.params(embedder_layer.embedders[2])[1]
            @test mapping_matrices[:Column3] == Flux.params(embedder_layer.embedders[3])[1]
            @test mapping_matrices[:Column6] == Flux.params(embedder_layer.embedders[6])[1]
            # dimensionalities are correct
            @test size(mapping_matrices[:Column2]) == expected_dims[3][1]
            @test size(mapping_matrices[:Column3]) == expected_dims[3][2]
            @test size(mapping_matrices[:Column6]) == expected_dims[3][3]
        end
    end
end
