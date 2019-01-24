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
using Printf
using Plots
plotly()

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
    collen = 20
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
        p = ordinalrank(Vector(aucs))
        for c in 1:2
            s = "$(aucs[c])"
            print(crays[p[c]], s)
            print(defc, repeat(" ", collen - length(s)) * " | ")
        end
        println(@sprintf("%.6f",df[i, 4]))
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
    averaged = []

    for b in unique(allData[:β])
        meanm1 = mean(allm1[allm1[:β] .== b, :][:auc])
        meanwass = mean(allwass[allwass[:β] .== b, :][:auc])
        push!(averaged, [meanm1, meanwass])
    end
    averaged = hcat(averaged...)
    pval = pvalue(SignedRankTest(averaged[1, :], averaged[2, :]))
    maxs = maximum(averaged, dims = 2)

    push!(compared, DataFrame(dataset = d, m1auc = maxs[1], wassauc = maxs[2], pval = pval))
end
compared = vcat(compared...)
CSV.write(dataFolder * "results-compared.csv", compared)
compared = CSV.read(dataFolder * "results-compared.csv")

sigdf = filter(x -> x[:pval] <= 0.05, compared)
diff = sigdf[:wassauc] .- sigdf[:m1auc]

pyplot()
plotlyjs()
p = plot(histogram(diff[diff .>= 0], bins = 0:0.08:0.8, seriescolor = "#1B9CE5", linecolor = false, xlabel = "difference in AUC (SVAE - VAE)", ylabel = "count", label = "SVAE"), size = (300, 300),  title = "VAE and SVAE comparison")
p = plot!(histogram!(diff[diff .< 0], bins = [-0.16, -0.08, 0], seriescolor = "#F51069", linecolor = false, label = "VAE"))
savefig(p, "figures/m1_vs_svae_hist.pdf")
