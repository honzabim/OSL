using JLD2
using FileIO
using DataFrames
using CSV
using UCI
using Statistics
using Crayons
using Crayons.Box
using StatsBase
using HypothesisTests

const dataFolder = "D:/dev/julia/OSL/experiments/SVAEvsM1pval/"

loadExperiment(filePath) = load(filePath)["results"]

datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "cardiotocography", "ecoli", "gisette", "glass", "haberman", "ionosphere", "iris", "isolet", "letter-recognition", "libras", "madelon", "magic-telescope", "miniboone", "multiple-features", "musk-2", "page-blocks", "pendigits", "pima-indians", "sonar", "spect", "statlog-satimage", "statlog-segment", "statlog-shuttle", "statlog-vehicle", "synthetic-control-chart", "vertebral-column", "wall-following-robot", "waveform-1", "waveform-2", "yeast", "wine"]
model = "svae"

params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :β, :method, :anomaliesSeen, :auc, :i, :model, :dataset]
types = [Int, Int, Int, String, String, Float64, String, Int, Float64, Int, String, String]

function processFile!(dataframe, model, dataset)
    for i in 1:100
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

function printbest2(df)
    collen = 15
    maxlen = maximum(vcat(length.(df[:dataset])..., 45))

    metds = ["m1", "svae-wass"]
    print(repeat(" ", maxlen) * " | ")
    for i in 1:2
        print(metds[i])
        print(repeat(" ", collen - length(metds[i])) * " | ")
    end
    println("pval")
    crays = [Crayon(foreground = :red), Crayon(foreground = :green)]
    defc = Crayon(reset = true)
    for i in 1:size(df, 1)
        print(defc, df[i, 1])
        print(defc, repeat(" ", maxlen - length(df[i, 1])) * " | ")
        aucs = df[i, 2:3]
        p = ordinalrank(vec(convert(Array, aucs)))
        for c in 1:2
            s = "$(aucs[c])"
            print(crays[p[c]], s)
            print(defc, repeat(" ", collen - length(s)) * " | ")
        end
        println(df[i, 4])
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

global compared = []
for d in unique(allData[:dataset])
    allm1 = allData[(allData[:method] .== "m1") .& (allData[:dataset] .== d), :]
    allwass = allData[(allData[:method] .== "wass") .& (allData[:dataset] .== d), :]

    iters = maximum(allm1[:i])
    maxed = []

    for i in 1:iters
        maxm1 = maximum(allm1[allm1[:i] .== i, :][:auc])
        maxwass = maximum(allwass[allwass[:i] .== i, :][:auc])
        push!(maxed, [maxm1, maxwass])
    end
    maxed = hcat(maxed...)
    pval = pvalue(SignedRankTest(maxed[1, :], maxed[2, :]))
    means = mean(maxed, 2)

    compared = push!(compared, DataFrame(dataset = d, m1auc = means[1], wassauc = means[2], pval = pval))
end
compared = vcat(compared...)
CSV.write(dataFolder * "results-compared.csv", compared)
