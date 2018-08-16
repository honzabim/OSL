using JLD2
using FileIO
using DataFrames
using CSV

# const dataFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2Latent/"
# const dataFolder = "d:/dev/data/loda/public/datasets/numerical/"
# const dataFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2LatentConsistency/"
# const dataFolder = "/home/jan/dev/OSL/experiments/firstVae/"
const dataFolder = "/home/jan/dev/OSL/experiments/SVAEkNN/"
# const dataFolder = "D:/dev/julia/OSL/experiments/WSVAE/"
# const datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
const datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
const models = ["svae"]
const scores = ["f1", "auc"]
const anomalycount = 5

loadExperiment(filePath) = load(filePath)["results"]

# params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :anomaliesSeen, :f1, :auc, :model, :dataset]
params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :β, :anomaliesSeen, :f1, :auc, :rsTrn, :rsTst, :knnauc, :knnprec, :knnrecall, :model, :dataset]
# types = [Int, Int, Int, String, String, Union{Int, Missings.Missing}, Union{Int, Missings.Missing}, Int, Float64, Float64, String, String]
types = [Int, Int, Int, String, String, Int, Int, Float64, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, String, String]
const anomalycount = 5
function processFile!(dataframe, model, dataset)
    println("Processing $model $dataset")
    results = loadExperiment(dataFolder * dataset * "-" * model * ".jld2")

    for i in 1:length(results)
        for ac in 1:anomalycount
            pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:7]..., results[i][1][9]) : vcat(results[i][1]..., -1, -1)
            # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
            push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., results[i][2][ac][6:10]..., model, dataset))
        end
    end
end

function ranks(allData, score)
    types = [String, Int]
    foreach(d -> push!(types, Int), models)

    names = [:dataset, :anomaliesSeen]
    foreach(x -> push!(names, Symbol(x)), models)

    rnks = DataFrame(types, names, 0)

    for ds in datasets
        for as in 1:anomalycount
            row = [ds, as]
            scores = map(model -> getMax(allData, ds, as, model, score), models)
            rnk = Array{Int}(length(models))
            for (i, j) in enumerate(sortperm(scores, rev = true))
                rnk[j] = i
            end
            foreach(i -> push!(row, i), rnk)
            push!(rnks, row)
        end
    end

    return rnks
end

getMax(allData, dataset, anomaliesSeen, model, score) = maximum(allData[(allData[:dataset] .== dataset) .* (allData[:model] .== model) .* (allData[:anomaliesSeen] .== anomaliesSeen), score])

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))
# for i in 1:100
#     processFile!(allData, models[1], datasets[1] * ".$i")
# end

CSV.write(dataFolder * "results.csv", allData)
#
# showall(ranks(allData, :f1))
# println()
# showall(ranks(allData, :auc))
# println()
#
# vaedf = allData[allData[:model] .== "autoencoder", :]
# # fmdf = allData[allData[:model] .== "ffMem", :]
# # ffdf = allData[allData[:model] .== "ff", :]
#
# by(allData, [:model, :dataset, :anomaliesSeen], (df) -> maximum(df[:f1]))
# println()
# by(allData, [:model, :dataset, :anomaliesSeen], (df) -> maximum(df[:auc]))
