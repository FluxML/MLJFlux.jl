function generate(dir; execute=true, pluto=false)
    quote
        using Pkg
        Pkg.activate(temp=true)
        Pkg.add("Literate")
        using Literate

        const OUTDIR = $dir
        const outdir = splitpath(OUTDIR)[end]
        const INFILE = joinpath(OUTDIR, "notebook.jl")

        @info "Generating notebooks for $outdir. "

        # generate pluto notebook:
        if $pluto
            TEMPDIR = tempdir()
            Literate.notebook(INFILE, TEMPDIR, flavor=Literate.PlutoFlavor())
            mv("$TEMPDIR/notebook.jl", "$OUTDIR/notebook.pluto.jl", force=true)
        else
            @warn "Not generating a Pluto notebook for $outdir."
        end

        Literate.notebook(INFILE, OUTDIR, execute=false)
        mv("$OUTDIR/notebook.ipynb", "$OUTDIR/notebook.unexecuted.ipynb", force=true)
        if $execute
            Literate.notebook(INFILE, OUTDIR, execute=true)
        else
            @warn "Not generating a pre-executed Jupyter notebook for $outdir. "
        end

        Literate.markdown(INFILE, OUTDIR)

    end |> eval
end

# Pkg.add("Pluto")
# using Pluto
# Pluto.run(notebook=joinpath(OUTDIR, "notebook.pluto.jl"))

# Pkg.add("IJulia")
# Pkg.instantiate()
# using IJulia
# IJulia.notebook(dir=OUTDIR)
# Pkg.add("IJulia")
# Pkg.instantiate()
# using IJulia
# IJulia.notebook(dir=OUTDIR)
