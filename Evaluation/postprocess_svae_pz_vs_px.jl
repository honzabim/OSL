using JLD2
using FileIO
using DataFrames
using Statistics
using CSV
using Crayons
using Crayons.Box
using StatsBase
using Printf

const dataFolder = "d:/dev/julia/OSL/experiments/WSVAE_PXV_vs_PZ/"
const datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "cardiotocography", "ecoli", "haberman", "ionosphere", "iris", "pima-indians", "sonar", "statlog-satimage", "waveform-1", "waveform-2", "wine", "yeast"]
const models = ["svae"]

loadExperiment(filePath) = load(filePath)["results"]

params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :β, :aucpxv, :aucpz, :ar, :i, :model, :dataset]
types = [Int, Int, Int, String, String, Float64, Float64, Float64, Float64, Int, String, String]

function processFile!(dataframe, model, dataset)
    for i in 1:10
        println("Processing $model $dataset $i")
        filename = dataFolder * dataset * "-$i-" * model * ".jld2"
        if !isfile(filename)
            println("$filename not found.")
        else
            results = loadExperiment(dataFolder * dataset * "-$i-" * model * ".jld2")

            for i in 1:length(results)
                for j in 1:length(results[1][2])
                    pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:6]...) : vcat(results[i][1]..., -1, -1)
                    # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                    push!(dataframe, vcat(pars..., results[i][2][j][1:4]..., model, dataset))
                end
            end
        end
    end
end

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))

CSV.write(dataFolder * "results.csv", allData)

function aggrmeanmax(df::DataFrame)
    dfagg = []
    for (ar, d) in Base.product(unique(df[:ar]), unique(df[:dataset]))
        dfaseen = df[(df[:dataset] .== d) .& (df[:ar] .== ar), :]
        parmean = []
        for b in unique(df[:β])
            dfperpars = dfaseen[dfaseen[:β] .== b, :]
            push!(parmean, [mean(dfperpars[:aucpxv]), mean(dfperpars[:aucpz])])
        end
        parmean = hcat(parmean...)
        maxad = maximum(parmean, dims = 2)
        push!(dfagg, DataFrame(dataset = d, anom_ratio = ar, aucpxv = maxad[1], aucpz = maxad[2]))
    end
    return vcat(dfagg...)
end

function printbest2(df)
    collen = 20
    maxlen = [maximum(vcat(length.(df[:dataset])..., 45)), 10]

    names = ["dataset", "anom_ratio"]
    for i in 1:2
        print(names[i])
        print(repeat(" ", maxlen[i] - length(names[i])) * " | ")
    end

    metds = ["pxv", "pz"]
    for i in 1:2
        print(metds[i])
        print(repeat(" ", collen - length(metds[i])) * " | ")
    end
    println()
    crays = [Crayon(foreground = :red), Crayon(foreground = :green)]
    defc = Crayon(reset = true)
    for i in 1:size(df, 1)
        for j in 1:2
            print(defc, df[i, j])
            print(defc, repeat(" ", maxlen[j] - length("$(df[i, j])")) * " | ")
        end
        aucs = df[i, 3:4]
        p = ordinalrank(Vector(aucs))
        for c in 1:2
            s = "$(aucs[c])"
            print(crays[p[c]], s)
            print(defc, repeat(" ", collen - length(s)) * " | ")
        end
        println()
    end
end
