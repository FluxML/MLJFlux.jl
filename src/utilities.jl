# # IMAGE COERCION

# Taken from ScientificTypes.jl to avoid as dependency.

_4Dcollection = AbstractArray{<:Real, 4}

function coerce(y::_4Dcollection, T2::Type{GrayImage})
    size(y, 3) == 1 || error("Multiple color channels encountered. "*
                      "Perhaps you want to use `coerce(image_collection, ColorImage)`.")
    y = dropdims(y, dims=3)
    return [ColorTypes.Gray.(y[:,:,idx]) for idx=1:size(y,3)]
end

function coerce(y::_4Dcollection, T2::Type{ColorImage})
    return [broadcast(ColorTypes.RGB, y[:,:,1, idx], y[:,:,2,idx], y[:,:,3, idx]) for idx=1:size(y,4)]
end


# # SYNTHETIC IMAGES

"""
    make_images(rng; image_size=(6, 6), n_classes=33, n_images=50, color=false, noise=0.05)

Return synthetic data of the form `(images, labels)` suitable for use
with MLJ's `ImageClassifier` model. All `images` are distortions of
`n_classes` fixed images. Two images with the same label correspond to
the same undistorted image.

"""
function make_images(rng; image_size=(6, 6), n_classes=33, n_images=50, color=false, noise=0.05)
    n_channels = color ? 3 : 1
    image_bag = map(1:n_classes) do _
        rand(rng, Float32, image_size...,  n_channels)
    end
    labels = rand(rng, 1:3, n_images)
    images = map(labels) do j
        image_bag[j] + noise*rand(rng, Float32, image_size..., n_channels)
    end
    T = color ? ColorImage : GrayImage
    X = coerce(cat(images...; dims=4), T)
    y = categorical(labels)
    return X, y
end
