#=

TODO: After https://github.com/FluxML/Metalhead.jl/issues/176:

- Export and externally document `metal` method

- Delete definition of `ResNetHack` below

- Change default builder in ImageClassifier (see /src/types.jl) from
  `metal(ResNetHack(...))` to `metal(Metalhead.ResNet(...))`,

- Add nicer `show` methods for `MetalheadBuilder` instances

=#

const DISALLOWED_KWARGS = [:imsize, :inchannels, :nclasses]
const human_disallowed_kwargs = join(map(s->"`$s`", DISALLOWED_KWARGS), ", ", " and ")
const ERR_METALHEAD_DISALLOWED_KWARGS = ArgumentError(
    "Keyword arguments $human_disallowed_kwargs are disallowed "*
    "as their values are inferred from data. "
)

# # WRAPPING

struct MetalheadWrapper{F} <: MLJFlux.Builder
    metalhead_constructor::F
end

struct MetalheadBuilder{F} <: MLJFlux.Builder
    metalhead_constructor::F
    args
    kwargs
end

Base.show(io::IO, w::MetalheadWrapper) =
    print(io, "metal($(repr(w.metalhead_constructor)))")

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
    print(io, "metal($(repr(w.metalhead_constructor)))(…)")


"""
    metal(constructor)(args...; kwargs...)

Return an MLJFlux builder object based on the Metalhead.jl constructor/type
`constructor` (eg, `Metalhead.ResNet`). Here `args` and `kwargs` are
passed to the `MetalheadType` constructor at "build time", along with
the extra keyword specifiers `imsize=...`, `inchannels=...` and
`nclasses=...`, with values inferred from the data.

# Example

If in Metalhead.jl you would do

```julia
using Metalhead
model = ResNet(50, pretrain=true, inchannels=1, nclasses=10)
```

then in MLJFlux, it suffices to do

```julia
using MLJFlux, Metalhead
builder = metal(ResNet)(50, pretrain=true)
```

which can be used in `ImageClassifier` as in

```julia
clf = ImageClassifier(
    builder=builder,
    epochs=500,
    optimiser=Flux.ADAM(0.001),
    loss=Flux.crossentropy,
    batch_size=5,
)
```

The keyord arguments `imsize`, `inchannels` and `nclasses` are
dissallowed in `kwargs` (see above).

"""
metal(metalhead_constructor) = MetalheadWrapper(metalhead_constructor)

function (pre_builder::MetalheadWrapper)(args...; kwargs...)
    kw_names = keys(kwargs)
    isempty(intersect(kw_names, DISALLOWED_KWARGS)) ||
        throw(ERR_METALHEAD_DISALLOWED_KWARGS)
    return MetalheadBuilder(pre_builder.metalhead_constructor, args, kwargs)
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
    imsize=nothing,
    inchannels=3,
    nclasses=1000,
    batchnorm=false,
    pretrain=false,
)

    # Note `imsize` is ignored, as here:
    # https://github.com/FluxML/Metalhead.jl/blob/9edff63222720ff84671b8087dd71eb370a6c35a/src/convnets/vgg.jl#L165

    @assert(
        depth in keys(Metalhead.vgg_config),
        "depth must be from one in $(sort(collect(keys(Metalhead.vgg_config))))"
    )
    model = Metalhead.VGG((224, 224);
                config = Metalhead.vgg_conv_config[Metalhead.vgg_config[depth]],
                inchannels,
                batchnorm,
                nclasses,
                fcsize = 4096,
                dropout = 0.5)
    if pretrain && !batchnorm
        Metalhead.loadpretrain!(model, string("VGG", depth))
    elseif pretrain
        Metalhead.loadpretrain!(model, "VGG$(depth)-BN)")
    end
    return model
end
