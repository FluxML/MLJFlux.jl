```@meta
EditURL = "<unknown>/mnist.jl"
```

# Using MLJ to classifiy the MNIST image dataset

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MLJ
using Flux
import MLJFlux
using Random
Random.seed!(123)

MLJ.color_off()

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));
nothing #hide
```

```
 Activating environment at `~/Dropbox/Julia7/MLJ/MLJFlux/examples/mnist/Project.toml`

```

## Basic training

Downloading the MNIST image dataset:

```julia
import Flux.Data.MNIST
images, labels = MNIST.images(), MNIST.labels();
nothing #hide
```

In MLJ, integers cannot be used for encoding categorical data, so we
must force the labels to have the `Multiclass` [scientific
type](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/). For
more on this, see [Working with Categorical
Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/).

```julia
labels = coerce(labels, Multiclass);
nothing #hide
```

Checking scientific types:

```julia
@assert scitype(images) <: AbstractVector{<:Image}
@assert scitype(labels) <: AbstractVector{<:Finite}
```

Looks good.

For general instructions on coercing image data, see [Type coercion
for image
data](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/#Type-coercion-for-image-data-1)

```julia
images[1]
```

```
28×28 Array{Gray{N0f8},2} with eltype Gray{FixedPointNumbers.Normed{UInt8,8}}:
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.012)  Gray{N0f8}(0.071)  Gray{N0f8}(0.071)  Gray{N0f8}(0.071)  Gray{N0f8}(0.494)  Gray{N0f8}(0.533)  Gray{N0f8}(0.686)  Gray{N0f8}(0.102)  Gray{N0f8}(0.651)  Gray{N0f8}(1.0)    Gray{N0f8}(0.969)  Gray{N0f8}(0.498)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.118)  Gray{N0f8}(0.141)  Gray{N0f8}(0.369)  Gray{N0f8}(0.604)  Gray{N0f8}(0.667)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.882)  Gray{N0f8}(0.675)  Gray{N0f8}(0.992)  Gray{N0f8}(0.949)  Gray{N0f8}(0.765)  Gray{N0f8}(0.251)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.192)  Gray{N0f8}(0.933)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.984)  Gray{N0f8}(0.365)  Gray{N0f8}(0.322)  Gray{N0f8}(0.322)  Gray{N0f8}(0.22)   Gray{N0f8}(0.153)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.071)  Gray{N0f8}(0.859)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.776)  Gray{N0f8}(0.714)  Gray{N0f8}(0.969)  Gray{N0f8}(0.945)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.314)  Gray{N0f8}(0.612)  Gray{N0f8}(0.42)   Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.804)  Gray{N0f8}(0.043)  Gray{N0f8}(0.0)    Gray{N0f8}(0.169)  Gray{N0f8}(0.604)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.055)  Gray{N0f8}(0.004)  Gray{N0f8}(0.604)  Gray{N0f8}(0.992)  Gray{N0f8}(0.353)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.545)  Gray{N0f8}(0.992)  Gray{N0f8}(0.745)  Gray{N0f8}(0.008)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.043)  Gray{N0f8}(0.745)  Gray{N0f8}(0.992)  Gray{N0f8}(0.275)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.137)  Gray{N0f8}(0.945)  Gray{N0f8}(0.882)  Gray{N0f8}(0.627)  Gray{N0f8}(0.424)  Gray{N0f8}(0.004)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.318)  Gray{N0f8}(0.941)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.467)  Gray{N0f8}(0.098)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.176)  Gray{N0f8}(0.729)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.588)  Gray{N0f8}(0.106)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.063)  Gray{N0f8}(0.365)  Gray{N0f8}(0.988)  Gray{N0f8}(0.992)  Gray{N0f8}(0.733)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.976)  Gray{N0f8}(0.992)  Gray{N0f8}(0.976)  Gray{N0f8}(0.251)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.18)   Gray{N0f8}(0.51)   Gray{N0f8}(0.718)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.812)  Gray{N0f8}(0.008)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.153)  Gray{N0f8}(0.58)   Gray{N0f8}(0.898)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.98)   Gray{N0f8}(0.714)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.094)  Gray{N0f8}(0.447)  Gray{N0f8}(0.867)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.788)  Gray{N0f8}(0.306)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.09)   Gray{N0f8}(0.259)  Gray{N0f8}(0.835)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.776)  Gray{N0f8}(0.318)  Gray{N0f8}(0.008)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.071)  Gray{N0f8}(0.671)  Gray{N0f8}(0.859)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.765)  Gray{N0f8}(0.314)  Gray{N0f8}(0.035)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.216)  Gray{N0f8}(0.675)  Gray{N0f8}(0.886)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.957)  Gray{N0f8}(0.522)  Gray{N0f8}(0.043)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.533)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.992)  Gray{N0f8}(0.831)  Gray{N0f8}(0.529)  Gray{N0f8}(0.518)  Gray{N0f8}(0.063)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
 Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)    Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)  Gray{N0f8}(0.0)
```

We start by defining a suitable `Builder` object. This is a recipe
for building the neural network. Our builder will work for images of
any (constant) size, whether they be color or black and white (ie,
single or multi-channel).  The architecture always consists of six
alternating convolution and max-pool layers, and a final dense
layer; the filter size and the number of channels after each
convolution layer is customisable.

```julia
import MLJFlux
mutable struct MyConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end

flatten(x::AbstractArray) = reshape(x, :, size(x)[end])
half(x) = div(x, 2)

function MLJFlux.build(b::MyConvBuilder, n_in, n_out, n_channels)

    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3

    mod(k, 2) == 1 || error("`filter_size` must be odd. ")

    p = div(k - 1, 2) # padding to preserve image size on convolution:

    h = n_in[1] |> half |> half |> half # final "image" height
    w = n_in[2] |> half |> half |> half # final "image" width

    return Chain(
        Conv((k, k), n_channels => c1, pad=(p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad=(p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad=(p, p), relu),
        MaxPool((2 ,2)),
        flatten,
        Dense(h*w*c3, n_out))
end
```

**Note.** There is no final `softmax` here, as this is applied by
default in all MLJFLux classifiers. Customisation of this behaviour
is controlled using using the `finaliser` hyperparameter of the
classifier.

We now define the MLJ model. If you have a GPU, substitute
`acceleration=CUDALibs()` below:

```julia
ImageClassifier = @load ImageClassifier
clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
                      acceleration=CPU1(),
                      batch_size=50,
                      epochs=10)
```

```
ImageClassifier(
    builder = Main.##403.MyConvBuilder(3, 16, 32, 32),
    finaliser = NNlib.softmax,
    optimiser = ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}()),
    loss = Flux.Losses.crossentropy,
    epochs = 10,
    batch_size = 50,
    lambda = 0.0,
    alpha = 0.0,
    optimiser_changes_trigger_retraining = false,
    acceleration = CPU1{Nothing}(nothing)) @688
```

You can add Flux options `optimiser=...` and `loss=...` here. At
present, `loss` must be a Flux-compatible loss, not an MLJ measure.

Binding the model with data in an MLJ machine:

```julia
mach = machine(clf, images, labels);
nothing #hide
```

Training for 10 epochs on the first 500 images:

```julia
fit!(mach, rows=1:500, verbosity=2);
nothing #hide
```

```
┌ Info: Training Machine{ImageClassifier{MyConvBuilder,…},…} @053.
└ @ MLJBase /Users/anthony/.julia/packages/MLJBase/4DmTL/src/machines.jl:341
┌ Info: Loss is 2.239
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 2.109
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 1.814
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 1.269
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 0.7602
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 0.5445
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 0.4606
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 0.341
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 0.2975
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122
┌ Info: Loss is 0.258
└ @ MLJFlux /Users/anthony/.julia/packages/MLJFlux/AWa8J/src/core.jl:122

```

Inspecting:

```julia
report(mach)
```

```
(training_losses = Float32[2.3228688, 2.2390091, 2.1091332, 1.8143247, 1.2688795, 0.7602044, 0.5444913, 0.46060604, 0.34104377, 0.29750612, 0.25796312],)
```

```julia
chain = fitted_params(mach)
```

```
(chain = Chain(Chain(Conv((3, 3), 1=>16, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((3, 3), 16=>32, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((3, 3), 32=>32, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), flatten, Dense(288, 10)), softmax),)
```

```julia
Flux.params(chain)[2]
```

```
16-element Array{Float32,1}:
  0.009390039
  0.07259901
 -0.0038282378
  0.016712788
  0.0019806586
  0.027747802
 -0.0007373232
  0.00018299253
  0.07081616
  0.06927007
  0.0020753173
  0.0032080673
  0.015448462
  0.008061458
  0.02398603
  0.047106728
```

Adding 20 more epochs:

```julia
clf.epochs = clf.epochs + 20
fit!(mach, rows=1:500);
nothing #hide
```

```
┌ Info: Updating Machine{ImageClassifier{MyConvBuilder,…},…} @053.
└ @ MLJBase /Users/anthony/.julia/packages/MLJBase/4DmTL/src/machines.jl:342
Optimising neural net:  3%[>                        ]  ETA: 0:00:00[KOptimising neural net:  6%[=>                       ]  ETA: 0:00:10[KOptimising neural net: 10%[==>                      ]  ETA: 0:00:12[KOptimising neural net: 13%[===>                     ]  ETA: 0:00:12[KOptimising neural net: 16%[====>                    ]  ETA: 0:00:12[KOptimising neural net: 19%[====>                    ]  ETA: 0:00:12[KOptimising neural net: 23%[=====>                   ]  ETA: 0:00:12[KOptimising neural net: 26%[======>                  ]  ETA: 0:00:12[KOptimising neural net: 29%[=======>                 ]  ETA: 0:00:11[KOptimising neural net: 32%[========>                ]  ETA: 0:00:11[KOptimising neural net: 35%[========>                ]  ETA: 0:00:10[KOptimising neural net: 39%[=========>               ]  ETA: 0:00:10[KOptimising neural net: 42%[==========>              ]  ETA: 0:00:10[KOptimising neural net: 45%[===========>             ]  ETA: 0:00:09[KOptimising neural net: 48%[============>            ]  ETA: 0:00:09[KOptimising neural net: 52%[============>            ]  ETA: 0:00:08[KOptimising neural net: 55%[=============>           ]  ETA: 0:00:08[KOptimising neural net: 58%[==============>          ]  ETA: 0:00:07[KOptimising neural net: 61%[===============>         ]  ETA: 0:00:07[KOptimising neural net: 65%[================>        ]  ETA: 0:00:06[KOptimising neural net: 68%[================>        ]  ETA: 0:00:06[KOptimising neural net: 71%[=================>       ]  ETA: 0:00:05[KOptimising neural net: 74%[==================>      ]  ETA: 0:00:05[KOptimising neural net: 77%[===================>     ]  ETA: 0:00:04[KOptimising neural net: 81%[====================>    ]  ETA: 0:00:03[KOptimising neural net: 84%[====================>    ]  ETA: 0:00:03[KOptimising neural net: 87%[=====================>   ]  ETA: 0:00:02[KOptimising neural net: 90%[======================>  ]  ETA: 0:00:02[KOptimising neural net: 94%[=======================> ]  ETA: 0:00:01[KOptimising neural net: 97%[========================>]  ETA: 0:00:01[KOptimising neural net:100%[=========================] Time: 0:00:18[K

```

Computing an out-of-sample estimate of the loss:

```julia
predicted_labels = predict(mach, rows=501:1000);
cross_entropy(predicted_labels, labels[501:1000]) |> mean
```

```
0.3694199f0
```

Or, in one line (after resetting the RNG seed to ensure the same
result):

```julia
Random.seed!(123)
evaluate!(mach,
          resampling=Holdout(fraction_train=0.5),
          measure=cross_entropy,
          rows=1:1000,
          verbosity=0)
```

```
┌───────────────────────┬───────────────┬────────────────┐
│ _.measure             │ _.measurement │ _.per_fold     │
├───────────────────────┼───────────────┼────────────────┤
│ LogLoss{Float64} @919 │ 0.366         │ Float32[0.366] │
└───────────────────────┴───────────────┴────────────────┘
_.per_observation = [[[6.14, 0.186, ..., 0.000432]]]
_.fitted_params_per_fold = [ … ]
_.report_per_fold = [ … ]

```

## Using out-of-sample loss estimates to terminate training:

MLJ will eventually provide model wrappers for controlling iterative
models. In the meantime some control can be implememted using the
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl)
package, and without the usual need for callbacks.

Defining an `EarlyStopper` object combining three separate stopping
critera:

```julia
using EarlyStopping
stopper = EarlyStopper(NotANumber(), Patience(3), UP())

losses = Float32[]
training_losses = Float32[];
nothing #hide
```

Resetting the number of epochs to zero:

```julia
clf.epochs = 0;
nothing #hide
```

Defining a function to increment the number of epochs, re-evaluate,
and test for early stopping:

```julia
function done()
    clf.epochs = clf.epochs + 1
    e = evaluate!(mach,
                  resampling=Holdout(fraction_train=0.5),
                  measure=cross_entropy,
                  rows=1:1000,
                  verbosity=0)
    loss = e.measurement[1][1]
    push!(losses, loss)
    training_loss = report(mach).training_losses[end]
    push!(training_losses, training_loss)
    println("out-of-sample loss: $loss")
    return done!(stopper, loss)
end;
nothing #hide
```

**Note.** Each time the number of epochs is increased and
`evaluate!` is called, warm-start training is used (assuming
`resampling isa Holdout`). This is because MLJ machines cache
hyper-parameters and learned parameters to avoid unnecessary
retraining. In other frameworks the same behaviour is implemented
using callbacks, but we don't need this here.

```julia
while !done() end
message(stopper)
```

```
"Early stop triggered by Patience(3) stopping criterion. "
```

A comparison of the training and out-of-sample losses:

```julia
plot(losses,
     title="Cross Entropy",
     xlab = "epoch",
     label="out-of-sample")
plot!(training_losses, label="training")
```
![](3192436122.png)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

