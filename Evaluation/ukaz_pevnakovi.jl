using CSV
using DataFrames

const dataPath = "../experiments/first_without_ff/results.csv"
const datasets = ["abalone", "breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
const models = ["autoencoder", "ffMem", "ff"]
const scores = ["f1", "auc"]
const anomalycount = 5


getMax(allData, dataset, anomaliesSeen, model, score) = maximum(allData[(allData[:dataset] .== dataset) .* (allData[:model] .== model) .* (allData[:anomaliesSeen] .== anomaliesSeen), score])

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

allData = CSV.read(dataPath)

showall(by(allData, [:model, :dataset, :anomaliesSeen], (df) -> maximum(df[:f1])))
println()
showall(by(allData, [:model, :dataset, :anomaliesSeen], (df) -> maximum(df[:auc])))
println()
showall(ranks(allData, :f1))
println()
showall(ranks(allData, :auc))
println()
