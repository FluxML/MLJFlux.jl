# # NEURAL NETWORK REGRESSOR

function shape(model::NeuralNetworkRegressor, X, y)
    n_input = Tables.schema(X).names |> length
    n_ouput = 1
    return (n_input, 1)
end

build(model::NeuralNetworkRegressor, shape) =
    build(model.builder, shape...)

fitresult(model::NeuralNetworkRegressor, chain, y) = (chain, nothing)

function MLJModelInterface.predict(model::NeuralNetworkRegressor,
                                   fitresult,
                                   Xnew)
    chain  = fitresult[1]
    Xnew_ = reformat(Xnew)
    return [chain(values.(tomat(Xnew_[:,i])))[1]
                for i in 1:size(Xnew_, 2)]
end

MLJModelInterface.metadata_model(NeuralNetworkRegressor,
               input=Table(Continuous),
               target=AbstractVector{<:Continuous},
               path="MLJFlux.NeuralNetworkRegressor",
               descr="A neural network model for making "*
                     "deterministic predictions of a "*
                     "`Continuous` target, given a table of "*
                     "`Continuous` features. ")


# # MULTITARGET NEURAL NETWORK REGRESSOR

function shape(model::MultitargetNeuralNetworkRegressor, X, y)
    n_input = Tables.schema(X).names |> length
    n_output = Tables.schema(y).names |> length
    return (n_input, n_output)
end

build(model::MultitargetNeuralNetworkRegressor, shape) =
    build(model.builder, shape...)

function fitresult(model::MultitargetNeuralNetworkRegressor, chain, y)
    target_column_names = Tables.schema(y).names
    return (chain, target_column_names)
end

function MLJModelInterface.predict(model::MultitargetNeuralNetworkRegressor,
                                   fitresult, Xnew)
    chain,  target_column_names = fitresult
    X = reformat(Xnew)
    ypred = [chain(values.(tomat(X[:,i])))
             for i in 1:size(X, 2)]
    return MLJModelInterface.table(reduce(hcat, y for y in ypred)',
                                       names=target_column_names)
end

MLJModelInterface.metadata_model(MultitargetNeuralNetworkRegressor,
               input=Table(Continuous),
               target=Table(Continuous),
               path="MLJFlux.MultitargetNeuralNetworkRegressor",
               descr = "A neural network model for making "*
                       "deterministic predictions of a "*
                       "`Continuous` multi-target, presented "*
                       "as a table, given a table of "*
                       "`Continuous` features. ")
