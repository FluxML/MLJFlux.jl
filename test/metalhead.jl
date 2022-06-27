using StableRNGs
using MLJFlux
const Metalhead = MLJFlux.Metalhead

@testset "display" begin
    io = IOBuffer()
    builder = MLJFlux.metal(MLJFlux.Metalhead.ResNet)(50, pretrain=false)
    show(io, MIME("text/plain"), builder)
    @test String(take!(io)) ==
        "builder wrapping Metalhead.ResNet\n  args:\n"*
        "    1: 50\n  kwargs:\n    pretrain = false\n"
    show(io, builder)
    @test String(take!(io)) == "metal(Metalhead.ResNet)(…)"
    close(io)
end

@testset "disallowed kwargs" begin
    @test_throws(
    MLJFlux.ERR_METALHEAD_DISALLOWED_KWARGS,
    MLJFlux.metal(MLJFlux.Metalhead.VGG)(imsize=(241, 241)),
    )
    @test_throws(
    MLJFlux.ERR_METALHEAD_DISALLOWED_KWARGS,
    MLJFlux.metal(MLJFlux.Metalhead.VGG)(inchannels=2),
    )
    @test_throws(
    MLJFlux.ERR_METALHEAD_DISALLOWED_KWARGS,
    MLJFlux.metal(MLJFlux.Metalhead.VGG)(nclasses=10),
    )
end

@testset "constructors" begin
    depth = 16
    imsize = (128, 128)
    nclasses = 10
    inchannels = 1
    wrapped = MLJFlux.metal(Metalhead.VGG)
    @test wrapped.metalhead_constructor == Metalhead.VGG
    builder = wrapped(depth, batchnorm=true)
    @test builder.metalhead_constructor == Metalhead.VGG
    @test builder.args == (depth, )
    @test (; builder.kwargs...) == (; batchnorm=true)
    ref_chain = Metalhead.VGG(
        imsize;
        config = Metalhead.vgg_conv_config[Metalhead.vgg_config[depth]],
        inchannels,
        batchnorm=true,
        nclasses,
        fcsize = 4096,
        dropout = 0.5
    )
    # needs https://github.com/FluxML/Metalhead.jl/issues/176
    # chain =
    #    MLJFlux.build(builder, StableRNGs.StableRNG(123), imsize, nclasses, inchannels)
    # @test length.(MLJFlux.Flux.params(ref_chain)) ==
    #    length.(MLJFlux.Flux.params(chain))
end

true
