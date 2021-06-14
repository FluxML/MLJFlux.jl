for Model in [:NeuralNetworkClassifier, :ImageClassifier]

    ex = quote
        mutable struct $Model{B,F,O,L} <: MLJModelInterface.Probabilistic
            builder::B
            finaliser::F
            optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
            loss::L        # can be called as in `loss(yhat, y)`
            epochs::Int    # number of epochs
            batch_size::Int  # size of a batch
            lambda::Float64  # regularization strength
            alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
            optimiser_changes_trigger_retraining::Bool
            acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
        end

        function $Model(; builder::B   = Short()
                        , finaliser::F = Flux.softmax
                        , optimiser::O = Flux.Optimise.ADAM()
                        , loss::L      = Flux.crossentropy
                        , epochs       = 10
                        , batch_size   = 1
                        , lambda       = 0
                        , alpha        = 0
                        , optimiser_changes_trigger_retraining = false
                        , acceleration = CPU1()
                        ) where {B,F,O,L}

            model = $Model{B,F,O,L}(builder
                                    , finaliser
                                    , optimiser
                                    , loss
                                    , epochs
                                    , batch_size
                                    , lambda
                                    , alpha
                                    , optimiser_changes_trigger_retraining
                                    , acceleration
                                    )

            message = clean!(model)
            isempty(message) || @warn message

            return model
        end
    end
    eval(ex)

end

for Model in [:NeuralNetworkRegressor, :MultitargetNeuralNetworkRegressor]

    ex = quote
        mutable struct $Model{B,O,L} <: MLJModelInterface.Deterministic
            builder::B
            optimiser::O  # mutable struct from Flux/src/optimise/optimisers.jl
            loss::L       # can be called as in `loss(yhat, y)`
            epochs::Int   # number of epochs
            batch_size::Int # size of a batch
            lambda::Float64 # regularization strength
            alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
            optimiser_changes_trigger_retraining::Bool
            acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
        end

        function $Model(; builder::B   = Linear()
                        , optimiser::O = Flux.Optimise.ADAM()
                        , loss::L      = Flux.mse
                        , epochs       = 10
                        , batch_size   = 1
                        , lambda       = 0
                        , alpha        = 0
                        , optimiser_changes_trigger_retraining=false
                        , acceleration  = CPU1()
                        ) where {B,O,L}

            model = $Model{B,O,L}(builder
                                  , optimiser
                                  , loss
                                  , epochs
                                  , batch_size
                                  , lambda
                                  , alpha
                                  , optimiser_changes_trigger_retraining
                                  , acceleration)

            message = clean!(model)
            isempty(message) || @warn message

            return model
        end
    end
    eval(ex)

end

const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}
