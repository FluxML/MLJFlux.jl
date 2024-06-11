# Execute this julia file to generate the notebooks from ../notebook.jl

joinpath(@__DIR__, "..", "generate.jl") |> include
generate(@__DIR__, execute=false, pluto=false)

