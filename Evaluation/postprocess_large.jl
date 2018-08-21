using JLD2
using FileIO
using DataFrames
using CSV

# const dataFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2Latent/"
# const dataFolder = "d:/dev/data/loda/public/datasets/numerical/"
# const dataFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2LatentConsistency/"
# const dataFolder = "/home/jan/dev/OSL/experiments/firstVae/"
# const dataFolder = "/home/jan/dev/OSL/experiments/WSVAElarge/"
const dataFolder = "D:/dev/julia/OSL/experiments/WSVAElarge/"
# const datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
# const datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# const datasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "glass", "ionosphere", "musk-2", "page-blocks", "pendigits", "pima-indians", "sonar", "statlog-satimage", "statlog-segment", "waveform-1", "waveform-2", "yeast"]
const datasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "glass", "ionosphere", "page-blocks", "pendigits", "pima-indians", "sonar", "statlog-satimage", "statlog-segment", "waveform-1", "waveform-2", "yeast"]
const models = ["svae"]
const anomalycount = 5

loadExperiment(filePath) = load(filePath)["results"]

# params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :anomaliesSeen, :f1, :auc, :model, :dataset]
params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :β, :anomaliesSeen, :f1, :auc, :subauc, :rsTrn, :rsTst, :knnauc, :knnprec, :knnrecall, :subknnauc, :subknnprec, :subknnrecall, :ar, :model, :dataset]
# types = [Int, Int, Int, String, String, Union{Int, Missings.Missing}, Union{Int, Missings.Missing}, Int, Float64, Float64, String, String]
types = [Int, Int, Int, String, String, Int, Int, Float64, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, String, String]
const anomalycount = 5
function processFile!(dataframe, model, dataset)
    println("Processing $model $dataset")
    results = loadExperiment(dataFolder * dataset * "-" * model * ".jld2")

    for i in 1:length(results)
        for j in 1:length(results[1][2])
            pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:7]..., results[i][1][9]) : vcat(results[i][1]..., -1, -1)
            # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
            push!(dataframe, vcat(pars..., results[i][2][j][1:3]..., results[i][2][j][6:15]..., model, dataset))
        end
    end
end

getMax(allData, dataset, anomaliesSeen, model, score) = maximum(allData[(allData[:dataset] .== dataset) .* (allData[:model] .== model) .* (allData[:anomaliesSeen] .== anomaliesSeen), score])

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))
# for i in 1:100
#     processFile!(allData, models[1], datasets[1] * ".$i")
# end

CSV.write(dataFolder * "results.csv", allData)
