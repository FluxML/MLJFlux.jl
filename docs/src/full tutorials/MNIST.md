## Image Classification Example
An expanded version of this example, with early stopping and
snapshots, is available [here](/examples/mnist).

We define a builder that builds a chain with six alternating
convolution and max-pool layers, and a final dense layer, which we
apply to the MNIST image dataset.

First we define a generic builder (working for any image size, color
or gray):

```julia
using MLJ
using Flux
using MLDatasets

# helper function
function flatten(x::AbstractArray)
	return reshape(x, :, size(x)[end])
end

import MLJFlux
mutable struct MyConvBuilder
	filter_size::Int
	channels1::Int
	channels2::Int
	channels3::Int
end

function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)

	k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3

	mod(k, 2) == 1 || error("`filter_size` must be odd. ")

	# padding to preserve image size on convolution:
	p = div(k - 1, 2)

	front = Chain(
			   Conv((k, k), n_channels => c1, pad=(p, p), relu),
			   MaxPool((2, 2)),
			   Conv((k, k), c1 => c2, pad=(p, p), relu),
			   MaxPool((2, 2)),
			   Conv((k, k), c2 => c3, pad=(p, p), relu),
			   MaxPool((2 ,2)),
			   flatten)
	d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
	return Chain(front, Dense(d, n_out))
end
```
Next, we load some of the MNIST data and check scientific types
conform to those is the table above:

```julia
N = 500
Xraw, yraw = MNIST.traindata();
Xraw = Xraw[:,:,1:N];
yraw = yraw[1:N];

scitype(Xraw)
```
```julia
scitype(yraw)
```

Inputs should have element scitype `GrayImage`:

```julia
X = coerce(Xraw, GrayImage);
```

For classifiers, target must have element scitype `<: Finite`:

```julia
y = coerce(yraw, Multiclass);
```

Instantiating an image classifier model:

```julia
ImageClassifier = @load ImageClassifier
clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
					  epochs=10,
					  loss=Flux.crossentropy)
```

And evaluating the accuracy of the model on a 30% holdout set:

```julia
mach = machine(clf, X, y)

evaluate!(mach,
				 resampling=Holdout(rng=123, fraction_train=0.7),
				 operation=predict_mode,
				 measure=misclassification_rate)
```
