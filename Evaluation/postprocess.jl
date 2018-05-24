using JLD2
using FileIO
using DataFrames
using CSV

const dataFolder = "/home/jan/dev/OSL/experiments/first_without_ff/"
const datasets = ["abalone", "breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
const models = ["autoencoder", "ffMem", "ff"]
const scores = ["f1", "auc"]

loadExperiment(filePath) = load(filePath)["results"]

params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :anomaliesSeen, :f1, :auc, :model, :dataset]
types = [Int, Int, Int, String, String, Union{Int, Missings.Missing}, Union{Int, Missings.Missing}, Int, Float64, Float64, String, String]
anomalycount = 5

function processFile!(dataframe, model, dataset)
    println("Processing $model $dataset")
    results = loadExperiment(dataFolder * dataset * "-" * model * ".jld2")

    for i in 1:length(results)
        for ac in 1:anomalycount
            pars = length(results[1][1]) > 5 ? results[i][1][1:7] : vcat(results[i][1]..., missing, missing)
            push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
        end
    end
end

function ranks(allData, score)
    types = [String]
    foreach(d -> push!(types, Int), models)

    names = [:dataset]
    foreach(x -> push!(names, Symbol(x)), models)

    rnks = DataFrame(types, names, 0)

    for ds in datasets
        row = Array{Any, 1}()
        push!(row, ds)
        scores = map(model -> getMax(allData, ds, model, score), models)
        rnk = Array{Int}(length(models))
        for (i, j) in enumerate(sortperm(scores, rev = true))
            rnk[j] = i
        end
        foreach(i -> push!(row, i), rnk)
        push!(rnks, row)
    end

    return rnks
end

getMax(allData, dataset, model, score) = maximum(allData[(allData[:dataset] .== dataset) .* (allData[:model] .== model), score])

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))
CSV.write(dataFolder * "results.csv", allData)

showall(ranks(allData, :f1))
showall(ranks(allData, :auc))
