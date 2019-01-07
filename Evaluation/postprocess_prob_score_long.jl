using JLD2
using FileIO
using DataFrames
using CSV

const dataFolder = "d:/dev/julia/OSL/experiments/WSVAElongProbMem/"
const datasets = ["blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "cardiotocography", "ecoli", "glass", "haberman", "ionosphere", "iris", "magic-telescope", "musk-2", "ionosphere", "page-blocks", "parkinsons", "pendigits", "pima-indians", "sonar", "spect-heart", "statlog-satimage", "statlog-vehicle", "synthetic-control-chart", "wall-following-robot", "waveform-1", "waveform-2", "wine", "yeast"]
const models = ["svae"]
const anomalycount = 10

loadExperiment(filePath) = load(filePath)["results"]


params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :β, :anomaliesSeen, :aucpxv, :aucf2, :aucf3, :ar, :i, :κ, :anom_sel, :model, :dataset]
types = [Int, Int, Int, String, String, Int, Int, Float64, Int, Float64, Float64, Float64, Float64, Int, Float64, String, String, String]

const anomalycount = 10
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
                    pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:7]..., results[i][1][9]) : vcat(results[i][1]..., -1, -1)
                    # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                    push!(dataframe, vcat(pars..., results[i][2][j][1:4]..., results[i][2][j][7:10]..., model, dataset))
                end
            end
        end
    end
end

getMax(allData, dataset, anomaliesSeen, model, score) = maximum(allData[(allData[:dataset] .== dataset) .* (allData[:model] .== model) .* (allData[:anomaliesSeen] .== anomaliesSeen), score])

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))

CSV.write(dataFolder * "results.csv", allData)

function aggr(df::DataFrame)
    dfagg = []
    for (d, asel, ar, aseen) in Base.product(unique(df[:dataset]), unique(df[:anom_sel]), unique(df[:ar]), unique(df[:anomaliesSeen]))
        dfall = df[(df[:dataset] .== d) .& (df[:anom_sel] .== asel) .& (df[:ar] .== ar) .& (df[:anomaliesSeen] .== aseen), :]
        itermax = []
        for i in 1:maximum(dfall[:i])
            dfperi = dfall[df[:i] .== i, :]
            push!(itermax, [maximum(dfperi[:aucpxv]), maximum(dfperi[:aucf2]), maximum(dfperi[:aucf3])])
        end
        itermax = hcat(itermax...)
        meanvals = mean(itermax, dims = 1)
        push!(dfagg, DataFrame(dataset = d, anom_sel = asel, anom_rati1o = ar, anom_seen = aseen, pxvita = meanvals[1], f2 = meanvals[2], f3 = meanvals[3]))
    end
    return vcat(dfagg...)
end
