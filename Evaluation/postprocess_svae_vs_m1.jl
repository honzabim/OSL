using JLD2
using FileIO
using DataFrames
using CSV
using UCI
using Statistics

const dataFolder = "D:/dev/julia/OSL/experiments/SVAEvsM1/"

loadExperiment(filePath) = load(filePath)["results"]

datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "cardiotocography", "ecoli", "gisette", "glass", "haberman", "ionosphere", "iris", "isolet", "letter-recognition", "libras", "madelon", "magic-telescope", "miniboone", "multiple-features", "musk-2", "page-blocks", "pendigits", "pima-indians", "sonar", "spect", "statlog-satimage", "statlog-segment", "statlog-shuttle", "statlog-vehicle", "synthetic-control-chart", "vertebral-column", "wall-following-robot", "waveform-1", "waveform-2", "yeast", "wine"]
model = "svae"

params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :β, :method, :anomaliesSeen, :auc, :i, :model, :dataset]
types = [Int, Int, Int, String, String, Float64, String, Int, Float64, Int, String, String]

function processFile!(dataframe, model, dataset)
    for i in 1:10
        println("Processing $model $dataset $i")
        filename = dataFolder * dataset * "-$i-" * model * ".jld2"
        if !isfile(filename)
            println("$filename not found.")
        else
            results = loadExperiment(filename)
            for i in 1:length(results)
                for j in 1:length(results[1][2])
                    pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:6]...) : vcat(results[i][1]..., -1, -1)
                    # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                    push!(dataframe, vcat(pars..., results[i][2][j][1:3]..., results[i][2][j][5]..., model, dataset))
                end
            end
        end
    end
end

allData = DataFrame(types, params, 0)
for d in datasets
    _, normal_labels, anomaly_labels = UCI.get_umap_data(d)
    if normal_labels == nothing
        processFile!(allData, model, d * " ")
    else
        for class in unique(anomaly_labels)
            processFile!(allData, model, d * " " * "$(normal_labels[1])-$(class)")
        end
    end
end

CSV.write(dataFolder * "results.csv", allData)

global averaged = DataFrame(types, params, 0)
for (m, d) in Base.product(unique(allData[:method]), unique(allData[:dataset]))
    newrow = allData[(allData[:method] .== m) .& (allData[:dataset] .== d) .& (allData[:i] .== 1), :]
    newrow[:auc] = mean(allData[(allData[:method] .== m) .& (allData[:dataset] .== d), :][:auc])
    averaged = vcat(averaged, newrow)
end
CSV.write(dataFolder * "results-avg.csv", averaged)

sumr = []
for d in unique(allData[:dataset])
    push!(sumr, DataFrame(dataset = d, m1auc = averaged[(averaged[:dataset] .== d) .& (averaged[:method] .== "m1"), :][:auc],
        lklhauc = averaged[(averaged[:dataset] .== d) .& (averaged[:method] .== "lklh"), :][:auc],
        wassauc = averaged[(averaged[:dataset] .== d) .& (averaged[:method] .== "wass"), :][:auc]))
end
sumr = vcat(sumr...)
