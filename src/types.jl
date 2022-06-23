abstract type MLJFluxProbabilistic <: MLJModelInterface.Probabilistic end
abstract type MLJFluxDeterministic <: MLJModelInterface.Deterministic end

const MLJFluxModel = Union{MLJFluxProbabilistic,MLJFluxDeterministic}

const doc_regressor(model_name) = """

    $model_name(; hyparameters...)

Instantiate an MLJFlux model. Available hyperparameters:

-  `builder`: Default = `MLJFlux.Linear(σ=Flux.relu)` (regressors) or
   `MLJFlux.Short(n_hidden=0, dropout=0.5, σ=Flux.σ)` (classifiers)

-  `optimiser`: The optimiser to use for training. Default =
   `Flux.ADAM()`

-  `loss`: The loss function used for training. Default = `Flux.mse`
   (regressors) and `Flux.crossentropy` (classifiers)

-  `epochs`: Number of epochs to train for. Default = `10`

-  `batch_size`: The batch_size for the data. Default = 1

-  `lambda`: The regularization strength. Default = 0. Range = [0, ∞)

-  `alpha`: The L2/L1 mix of regularization. Default = 0. Range = [0, 1]

-  `rng`: The random number generator (RNG) passed to builders, for
   weight intitialization, for example. Can be any `AbstractRNG` or
   the seed (integer) for a `MersenneTwister` that is reset on every
   cold restart of model (machine) training. Default =
   `GLOBAL_RNG`.

-  `acceleration`: Use `CUDALibs()` for training on GPU; default is `CPU1()`.

- `optimiser_changes_trigger_retraining`: True if fitting an
   associated machine should trigger retraining from scratch whenever
   the optimiser changes. Default = `false`

"""

doc_classifier(model_name) = doc_regressor(model_name)*"""
- `finaliser`: Operation applied to the unnormalized output of the
  final layer to obtain probabilities (outputs summing to
  one). The shape of the inputs and outputs
  of this operator must match.  Default = `Flux.softmax`.

"""

for Model in [:NeuralNetworkClassifier, :ImageClassifier]

    default_builder_ex = Model == :ImageClassifier ? :(metal(VGGHack)()) : Short()

    ex = quote
        mutable struct $Model{B,F,O,L} <: MLJFluxProbabilistic
            builder::B
            finaliser::F
            optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
            loss::L        # can be called as in `loss(yhat, y)`
            epochs::Int    # number of epochs
            batch_size::Int  # size of a batch
            lambda::Float64  # regularization strength
            alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
            rng::Union{AbstractRNG,Int64}
            optimiser_changes_trigger_retraining::Bool
            acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
        end

        function $Model(; builder::B   = $default_builder_ex
                        , finaliser::F = Flux.softmax
                        , optimiser::O = Flux.Optimise.ADAM()
                        , loss::L      = Flux.crossentropy
                        , epochs       = 10
                        , batch_size   = 1
                        , lambda       = 0
                        , alpha        = 0
                        , rng          = Random.GLOBAL_RNG
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
                                    , rng
                                    , optimiser_changes_trigger_retraining
                                    , acceleration
                                    )

            message = clean!(model)
            isempty(message) || @warn message

            return model
        end

        @doc doc_classifier($Model) $Model

    end
    eval(ex)

end

for Model in [:NeuralNetworkRegressor, :MultitargetNeuralNetworkRegressor]

    ex = quote
        mutable struct $Model{B,O,L} <: MLJFluxDeterministic
            builder::B
            optimiser::O  # mutable struct from Flux/src/optimise/optimisers.jl
            loss::L       # can be called as in `loss(yhat, y)`
            epochs::Int   # number of epochs
            batch_size::Int # size of a batch
            lambda::Float64 # regularization strength
            alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
            rng::Union{AbstractRNG,Integer}
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
                        , rng          = Random.GLOBAL_RNG
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
                                  , rng
                                  , optimiser_changes_trigger_retraining
                                  , acceleration)

            message = clean!(model)
            isempty(message) || @warn message

            return model
        end

        @doc $doc_regressor($Model) $Model

    end
    eval(ex)

end

const Regressor =
    Union{NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor}
