using RDatasets
using MLJFlux
using MLJ
using Flux
using CategoricalArrays
using MLJBase
data = dataset("mlmRev", "Exam")

X = data[[:School, :SchGend, :VR, :Intake, :Sex, :Type, :Student]]

y = data[:NormExam]

struct mynn <: MLJFlux.Builder
    d1
    d2
    d3
end

function MLJFlux.fit(nn::mynn, ip, op)
    return Chain(Dense(ip, nn.d1, sigmoid), Dense(nn.d1, nn.d2, relu), Dense(nn.d2, nn.d3, sigmoid), Dense(nn.d3, op))
end

train, test = partition(eachindex(y), 0.75);

nn = mynn(128, 64, 32)

regressor = NeuralNetworkRegressor(builder=nn, optimiser=ADAM(0.0001), batch_size=20, embedding_choice=:entity_embedding, n=30)

mach = machine(regressor, X, y)

fit!(mach, verbosity=2)

yhat = predict(mach, rows=test)

ee_loss = regressor.loss(getindex.(getindex.(yhat, 1), 1), y[test])
println("Loss with entity embedings is $ee_loss")

onehot_model = NeuralNetworkRegressor(builder=nn, optimiser=ADAM(0.0001), batch_size=20, n=30)

mach2 = machine(onehot_model, X, y)

fit!(mach2, verbosity=2)

yhat = predict(mach2, rows=test)

onehot_loss = regressor.loss(getindex.(getindex.(yhat, 1), 1), y[test])
println("Loss with entity embedings is $onehot_loss")
