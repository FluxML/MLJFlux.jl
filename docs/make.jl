using Documenter
using MLJFlux

DocMeta.setdocmeta!(MLJFlux, :DocTestSetup, :(using MLJFlux); recursive=true)

makedocs(
	sitename = "MLJFlux",
    format = Documenter.HTML(;
    assets = [
        "assets/favicon.ico",
        asset(
            "https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap",
            class = :css,
        ),
    ],
    repolink="https://github.com/FluxML/MLJFlux.jl"
),
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
