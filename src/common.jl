MLJFluxModel = Union{NeuralNetworkRegressor,
                     MultitargetNeuralNetworkRegressor,
                     NeuralNetworkClassifier,
                     ImageClassifier}

function MLJModelInterface.clean!(model::MLJFluxModel)
    warning = ""
    if model.lambda < 0
        warning *= "Need `lambda ≥ 0`. Resetting `lambda = 0`. "
        model.lambda = 0
    end
    if model.alpha < 0 || model.alpha > 1
        warning *= "Need alpha in the interval `[0, 1]`. Resetting `alpha = 0`. "
        model.alpha = 0
    end
    if model.epochs < 0 
        warning *= "Need `epochs ≥ 0`. Resetting `epochs = 10`. "
        model.epochs = 10
    end
    if model.batch_size < 0 
        warning *= "Need `batch_size ≥ 0`. Resetting `batch_size = 1`. "
        model.batch_size = 1
    end
    if model.acceleration isa CUDALibs && gpu_isdead()
        warning *= "`acceleration isa CUDALibs` "*
            "but no CUDA device (GPU) currently live. "
    end
    return warning
end
