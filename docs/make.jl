using Documenter
using MLJFlux
using Flux

DocMeta.setdocmeta!(MLJFlux, :DocTestSetup, :(using MLJFlux); recursive=true)

makedocs(
    sitename = "MLJFlux",
    format = Documenter.HTML(
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(
                "https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap",
                class = :css,
            ),
            asset(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
                class = :css,
            )
        ],
        repolink="https://github.com/FluxML/MLJFlux.jl"
    ),
    modules = [MLJFlux],
    warnonly = true,
    pages = [
        "Introduction" => "index.md",
        "Interface"=> Any[
            "Summary"=>"interface/Summary.md",
            "Builders"=>"interface/Builders.md",
            "Custom Builders"=>"interface/Custom Builders.md",
            "Classification"=>"interface/Classification.md",
            "Regression"=>"interface/Regression.md",
            "Multi-Target Regression"=>"interface/Multitarget Regression.md",
            "Image Classification"=>"interface/Image Classification.md",
        ],
        "Workflow Examples" => Any[
            "Incremental Training"=>
                "workflow_examples/Incremental Training/notebook.md",
            "Hyperparameter Tuning"=>
                "workflow_examples/Hyperparameter Tuning/notebook.md",
            "Neural Architecture Search"=>
                "workflow_examples/Basic Neural Architecture Search/notebook.md",
            "Model Composition"=>"workflow_examples/Composition/notebook.md",
            "Model Comparison"=>"workflow_examples/Comparison/notebook.md",
            "Early Stopping"=>"workflow_examples/Early Stopping/notebook.md",
            "Live Training"=>"workflow_examples/Live Training/notebook.md",
        ],
        # "Tutorials"=>Any[
        # "Spam Detection with RNNs"=>"extended_examples/Spam Detection with RNNs/notebook.md"
        # ],
        "Contributing" => "contributing.md"],
    doctest = false,
)

# Documenter can also automatically deploy documentation to gh-pages.
deploydocs(repo = "github.com/FluxML/MLJFlux.jl.git", devbranch = "dev")
