using Documenter
using MLJFlux

DocMeta.setdocmeta!(MLJFlux, :DocTestSetup, :(using MLJFlux); recursive=true)

makedocs(
	sitename = "MLJFlux",
	format = Documenter.HTML(repolink="https://github.com/FluxML/MLJFlux.jl"),
	modules = [MLJFlux],
	warnonly = true,
	pages = ["Introduction" => "index.md",
            "API"=> "api.md",
            "Features" => Any[
                "Tuning"=>"features/tuning.md",
                "Early Stopping"=>"features/early.md",
            ],
            "Contributing" => "contributing.md",
            "About" => "about.md"],
        doctest = false,
)

# Documenter can also automatically deploy documentation to gh-pages.
deploydocs(repo = "github.com/FluxML/MLJFlux.jl.git", devbranch = "dev")
