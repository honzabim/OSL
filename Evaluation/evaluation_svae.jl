using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern
using JLD2
using FileIO

folderpath = "/home/jan/dev/"
# folderpath = "D:/dev/"
push!(LOAD_PATH, folderpath * "OSL/KNNmemory", folderpath * "FluxExtensions.jl/src", folderpath * "anomaly detection/anomaly_detection/src", folderpath * "EvalCurves.jl/src", folderpath)
using KNNmem
using FluxExtensions
using AnomalyDetection
using EvalCurves
using ADatasets

include(folderpath * "OSL/SVAE/svae.jl")

# TODO export neco, necoJinyho

"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function createSVAEWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, T = Float32)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE(encoder, decoder, hiddenDim, latentDim, T)
    train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        trainOnLatent!(zfromx(svae, data), zeros(collect(labels))) # changes labels to zeros!
        return loss(svae, data)
    end

    function learnAnomaly!(data, labels)
        trainOnLatent!(zfromx(svae, data), labels)
        return loss(svae, data)
    end

    return svae, learnRepresentation!, learnAnomaly!, classify
end

function rscore(m::SVAE, x)
    xgivenz = m.g(zfromx(m, x))
    return Flux.mse(x, xgivenz)
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000)
    (model, learnRepresentation!, learnAnomaly!, classify) = createModel()
    opt = Flux.Optimise.ADAM(Flux.params(model))
    FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2] .- 1), batchSize, numBatches), ()->(), 1000)

    learnRepresentation!(train[1], train[2] - 1)

    rstrn = Flux.Tracker.data(rscore(model, train[1]))
    rstst = Flux.Tracker.data(rscore(model, test[1]))

    results = []
    anomalies = train[1][:, train[2] .- 1 .== 1] # TODO needs to be shuffled!!!
    for ac in anomalyCounts
        if ac <= size(anomalies, 2)
            l = learnAnomaly!(anomalies[:, ac], [1])
        else
            break;
        end

        values, probScore = classify(test[1])
        values = Flux.Tracker.data(values)
        probScore = Flux.Tracker.data(probScore)

        rocData = roc(test[2] .- 1, values)
        f1 = f1score(rocData)
        auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
        push!(results, (ac, f1, auc, values, probScore, rstrn, rstst))
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/SVAEaftertraining/"
mkpath(outputFolder)

datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
difficulties = ["easy", "easy", "easy", "easy"]
const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 10000

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "high")

for (dn, df) in zip(datasets, difficulties)
    train, test, clusterdness = loadData(dn, df)

    println("$dn")
    println("Running svae...")

    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), 1:10, batchSize, iterations)
    results = gridSearch(evaluateOneConfig, [32], [4 8 16], [3], ["leakyrelu"], ["Dense"], [1024], [64], 1)
    results = reshape(results, length(results), 1)
    save(outputFolder * dn * "-svae.jld2", "results", results)
end
