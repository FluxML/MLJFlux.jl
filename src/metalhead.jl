#=

TODO: After https://github.com/FluxML/Metalhead.jl/issues/176:

- Export and externally document `image_builder` method

- Delete definition of `VGGHack` below

- Change default builder in ImageClassifier (see /src/types.jl) from
  `image_builder(VGGHack)` to `image_builder(Metalhead.VGG)`.

=#

const DISALLOWED_KWARGS = [:imsize, :inchannels, :nclasses]
const human_disallowed_kwargs = join(map(s->"`$s`", DISALLOWED_KWARGS), ", ", " and ")
const ERR_METALHEAD_DISALLOWED_KWARGS = ArgumentError(
    "Keyword arguments $human_disallowed_kwargs are disallowed "*
    "as their values are inferred from data. "
)

# # WRAPPING

struct MetalheadBuilder{F} <: MLJFlux.Builder
    metalhead_constructor::F
    args
    kwargs
end

function Base.show(io::IO, ::MIME"text/plain", w::MetalheadBuilder)
    println(io, "builder wrapping $(w.metalhead_constructor)")
    if !isempty(w.args)
        println(io, "  args:")
        for (i, arg) in enumerate(w.args)
            println(io, "    1: $arg")
        end
    end
    if !isempty(w.kwargs)
        println(io, "  kwargs:")
        for kwarg in w.kwargs
            println(io, "    $(first(kwarg)) = $(last(kwarg))")
        end
    end
end

Base.show(io::IO, w::MetalheadBuilder) =
    print(io, "image_builder($(repr(w.metalhead_constructor)), …)")


"""
    image_builder(metalhead_constructor, args...; kwargs...)

Return an MLJFlux builder object based on the Metalhead.jl constructor/type
`metalhead_constructor` (eg, `Metalhead.ResNet`). Here `args` and `kwargs` are
passed as arguments to `metalhead_constructor` at "build time", along with
the extra keyword specifiers `imsize=...`, `inchannels=...` and
`nclasses=...`, with values inferred from the data.

# Example

If in Metalhead.jl you would do

```julia
using Metalhead
model = ResNet(50, pretrain=false, inchannels=1, nclasses=10)
```

then in MLJFlux, it suffices to do

```julia
using MLJFlux, Metalhead
builder = image_builder(ResNet, 50, pretrain=false)
```

which can be used in `ImageClassifier` as in

```julia
clf = ImageClassifier(
    builder=builder,
    epochs=500,
    optimiser=Flux.Adam(0.001),
    loss=Flux.crossentropy,
    batch_size=5,
)
```

The keyord arguments `imsize`, `inchannels` and `nclasses` are
dissallowed in `kwargs` (see above).

"""
function image_builder(
    metalhead_constructor,
    args...;
    kwargs...
)
    kw_names = keys(kwargs)
    isempty(intersect(kw_names, DISALLOWED_KWARGS)) ||
        throw(ERR_METALHEAD_DISALLOWED_KWARGS)
    return MetalheadBuilder(metalhead_constructor, args, kwargs)
end

MLJFlux.build(
    b::MetalheadBuilder,
    rng,
    n_in,
    n_out,
    n_channels
) =  b.metalhead_constructor(
    b.args...;
    b.kwargs...,
    imsize=n_in,
    inchannels=n_channels,
    nclasses=n_out
)

# See above "TODO" list.
function VGGHack(
    depth::Integer=16;
    imsize=(242,242),
    inchannels=3,
    nclasses=1000,
    batchnorm=false,
    pretrain=false,
)

    # Adapted from
    # https://github.com/FluxML/Metalhead.jl/blob/4e5b8f16964468518eeb6eb8d7e5f85af4ecf959/src/convnets/vgg.jl#L161
    # But we do not ignore `imsize`.

    @assert(
        depth in keys(Metalhead.VGG_CONFIGS),
        "depth must be from one in $(sort(collect(keys(Metalhead.VGG_CONFIGS))))"
    )
    model = Metalhead.VGG(imsize;
                config = Metalhead.VGG_CONFIGS[depth],
                inchannels,
                batchnorm,
                nclasses,
                dropout_prob = 0.5)
    if pretrain
        imsize == (224, 224) || @warn "Using `pretrain=true` may not work unless "*
            "image size is `(224, 224)`, which it is not. "
        artifact_name = string("vgg", depth)
        if batchnorm
            artifact_name *= "_bn"
        else
            artifact_name *= "-IMAGENET1K_V1"
        end
        loadpretrain!(model, artifact_name)
    end

    return model
end
