using MLJ
using MLJFlux

## Test 1: Data with purely categorical features, onehot encoding and batch size =1

nn = NeuralNetworkRegressor(builder = MLJFlux.Short())
x, y = load_reduced_ames()();
x = x[[:MSSubClass, :Neighborhood]]

mach = machine(nn, x, y)

fit!(mach, verbosity=2)

## Test 2 : Same confuguration as above, but batch_size  > 1
nn.batch_size = 3
fit!(mach, verbosity=2)

nn.embedding_choice = :entity_embedding
fit!(mach, verbosity=2)

## Test 3: EntityEmbeddings with purely categorical features

nn2 = NeuralNetworkRegressor(builder= MLJFlux.Short(), embedding_choice=:entity_embedding)

mach2 = machine(nn2, x, y)

fit!(mach2)

## Test 4: EE with batch_size > 1

nn2.batch_size = 3

fit!(mach2)

nn2.embedding_choice = :onehot
fit!(mach2)


## NNClassifiers

using DataFrames
using CategoricalArrays
using Flux

#ip = (x = rand("abcd", 100), y = rand("edf", 100), z=rand("hijk", 100)) |> MLJBase.table;
op = rand("1234", 1456) |> CategoricalArray

nnclassifier = NeuralNetworkClassifier(builder=MLJFlux.Short(), loss=Flux.mse, batch_size=2, n = 3)

nn_mach = machine(nnclassifier, x, op)
fit!(nn_mach)

nnclassifier.batch_size = 3
d = fit!(nn_mach)

nnclassifier.embedding_choice = :entity_embedding

fit!(nn_mach)
