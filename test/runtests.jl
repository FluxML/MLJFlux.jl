# using Revise
import MLJBase
import FluxMLJ.MLJInterface
const Intf = MLJInterface

# in MLJ multivariate inputs are tables:
X = MLJBase.table(randn(20, 5))

# while multivariate targets are vectors of tuples:
ymatrix = hcat(1 .+ X.x1 - X.x2, 1 .- 2X.x4 + X.x5)
y = [Tuple(ymatrix[i,:]) for i in 1:size(ymatrix, 1)]

model = Intf.NeuralNetworkRegressor()
fitresult, cache, report = MLJBase.fit(model, 1, X, y)
