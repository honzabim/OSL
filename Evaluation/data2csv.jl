using ADatasets
using CSV
using DataFrames

# const folderpath = "/home/bimjan/dev/julia/"
const folderpath = "D:/dev/julia/"
const dataPath = folderpath * "data/loda/public/datasets/numerical"

dataset = ARGS[1]
difficulty = "easy"
anomalyRatios = [0.05, 0.01, 0.005]
iterations = 1:10

outputFolder = folderpath * "data/CSV/" * dataset
mkpath(outputFolder)

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

function savetodf(ds)
    header = vcat("label", (repeat(["X"], size(ds[1], 1)) .* string.(collect(1:size(ds[1], 1)))...))
    labels = (x -> x == 1 ? "nominal" : "anomaly").(ds[2])
    data = hcat(labels, collect(ds[1]'))
    df = DataFrame(data)
    rename!(df, :x1, :label)
    return df
end

for i in iterations
    trainall, test, clusterdness = loadData(dataset, difficulty)
    if i == 1
        CSV.write(outputFolder * "/" * dataset * "-test.csv", savetodf(test))
    end
    for ar in anomalyRatios
        train = ADatasets.subsampleanomalous(trainall, ar)
        CSV.write(outputFolder * "/" * dataset * "-train-$i-$ar.csv", savetodf(train))
    end
end
