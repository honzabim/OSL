using JLD2
using FileIO
using DataFrames
using CSV

# const dataFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2Latent/"
# const dataFolder = "d:/dev/data/loda/public/datasets/numerical/"
# const dataFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2LatentConsistency/"
# const dataFolder = "/home/jan/dev/OSL/experiments/firstVae/"
# const dataFolder = "/home/jan/dev/OSL/experiments/WSVAElarge/"
const dataFolder = "D:/dev/julia/OSL/experiments/WSVAElargeVarOfDistances/"
# const datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
# const datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# const datasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "glass", "ionosphere", "musk-2", "page-blocks", "pendigits", "pima-indians", "sonar", "statlog-satimage", "statlog-segment", "waveform-1", "waveform-2", "yeast"]
# const datasets = ["cardiotocography", "ecoli", "glass", "statlog-segment", "yeast"]
# const datasets = ["cardiotocography", "ecoli", "glass", "pendigits", "pima-indians", "statlog-satimage", "statlog-segment", "waveform-1"]
# const datasets = ["breast-cancer-wisconsin", "ecoli", "glass", "ionosphere", "pendigits", "pima-indians", "statlog-satimage", "statlog-segment", "waveform-1", "waveform-2", "yeast"]
const datasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "glass", "musk-2", "ionosphere", "page-blocks", "pendigits", "pima-indians", "sonar", "statlog-satimage", "statlog-segment", "waveform-1", "waveform-2", "yeast"]
# const datasets = ["cardiotocography", "ecoli", "glass", "musk-2", "ionosphere", "page-blocks", "pendigits", "pima-indians", "sonar", "statlog-satimage", "statlog-segment", "waveform-1", "waveform-2", "yeast"]
const models = ["svae"]
const anomalycount = 5

loadExperiment(filePath) = load(filePath)["results"]

# params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :anomaliesSeen, :f1, :auc, :model, :dataset]
params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :β, :anomaliesSeen, :f1, :auc, :rsTrn, :rsTst, :knnauc, :knnprec, :knnrecall, :ar, :i, :model, :dataset, :mpwmutualinf, :varOfDists, :knn5auc, :knn9auc, :knn15auc, :knnsqrtauc, :knn5a3auc, :knn5a5auc, :knn5a9auc, :knn5asqrtauc]
# types = [Int, Int, Int, String, String, Union{Int, Missings.Missing}, Union{Int, Missings.Missing}, Int, Float64, Float64, String, String]
types = [Int, Int, Int, String, String, Int, Int, Float64, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int, String, String, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]
const anomalycount = 5
function processFile!(dataframe, model, dataset)
    for i in 1:10
        println("Processing $model $dataset $i")
        results = loadExperiment(dataFolder * dataset * "-$i-" * model * ".jld2")

        for i in 1:length(results)
            for j in 1:length(results[1][2])
                pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:7]..., results[i][1][9]) : vcat(results[i][1]..., -1, -1)
                # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                println(length(results[1][2][j]))
                push!(dataframe, vcat(pars..., results[i][2][j][1:3]..., results[i][2][j][6:12]..., model, dataset, results[i][2][j][13:22]...))
            end
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

function getbestknn(df)
	maxknn = []
	for d in unique(df[:dataset])
		dfknn = df[(df[:dataset] .== d) .& (df[:ar] .== 0.05), :][23:26]
		best = maximum(convert(Array, dfknn))
		push!(maxknn, DataFrame(dataset = d, bestknn = best))
	end
	return vcat(maxknn...)
end

function compare(df)
	compdf = []
	for (ar, d) in Base.product(unique(df[:ar]), unique(df[:dataset]))
		dfda = df[(df[:dataset] .== d) .& (df[:ar] .== ar), :]
		bestknn = maximum(convert(Array, dfda[23:26]))
		bestknna5 = maximum(convert(Array, dfda[27:30]))
		bestsvaemem = maximum(convert(Array, dfda[:auc]))
		push!(compdf, DataFrame(dataset = d, anom_ratio = ar, svae_mem = bestsvaemem, knn = bestknn, knn5a = bestknna5))
	end
	return vcat(compdf...)
end
