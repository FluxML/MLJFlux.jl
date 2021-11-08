# Execute this julia file to generate the notebooks from ../notebook.jl

env = @__DIR__
joinpath(env, "..", "..", "generate.jl") |> include
generate(env, execute=false, pluto=false)
