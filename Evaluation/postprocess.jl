using JLD2
using FileIO
using DataFrames

const dataFolder = "/home/jan/dev/OSL/experiments/test/"
const datasets = ["abalone", "breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
const models = ["autoencoder", "ffMem", "ff"]

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

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))
