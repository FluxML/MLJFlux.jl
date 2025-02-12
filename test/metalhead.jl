using StableRNGs
using MLJFlux
const Metalhead = MLJFlux.Metalhead

@testset "display" begin
    io = IOBuffer()
    builder = MLJFlux.image_builder(MLJFlux.Metalhead.ResNet, 50, pretrain=false)
    show(io, MIME("text/plain"), builder)
    @test String(take!(io)) ==
        "builder wrapping Metalhead.ResNet\n  args:\n"*
        "    1: 50\n  kwargs:\n    pretrain = false\n"
    show(io, builder)
    @test String(take!(io)) == "image_builder(Metalhead.ResNet, …)"
    close(io)
end

@testset "disallowed kwargs" begin
    @test_throws(
    MLJFlux.ERR_METALHEAD_DISALLOWED_KWARGS,
    MLJFlux.image_builder(MLJFlux.Metalhead.VGG, imsize=(241, 241)),
    )
    @test_throws(
    MLJFlux.ERR_METALHEAD_DISALLOWED_KWARGS,
    MLJFlux.image_builder(MLJFlux.Metalhead.VGG, inchannels=2),
    )
    @test_throws(
    MLJFlux.ERR_METALHEAD_DISALLOWED_KWARGS,
    MLJFlux.image_builder(MLJFlux.Metalhead.VGG, nclasses=10),
    )
end

@testset "constructors" begin
    depth = 16
    imsize = (128, 128)
    nclasses = 10
    inchannels = 1
    builder = MLJFlux.image_builder(
        Metalhead.VGG,
        depth,
        batchnorm=true
    )
    @test builder.metalhead_constructor == Metalhead.VGG
    @test builder.args == (depth, )
    @test (; builder.kwargs...) == (; batchnorm=true)

    ## needs https://github.com/FluxML/Metalhead.jl/issues/176:
    # ref_chain = Metalhead.VGG(
    #     imsize;
    #     config = Metalhead.VGG_CONV_CONFIGS[Metalhead.VGG_CONFIGS[depth]],
    #     inchannels,
    #     batchnorm=true,
    #     nclasses,
    # )
    # chain =
    #    MLJFlux.build(builder, StableRNGs.StableRNG(123), imsize, nclasses, inchannels)
    # @test length.(MLJFlux.Flux.trainables(ref_chain)) ==
    #    length.(MLJFlux.Flux.trainables(chain))
end

true
