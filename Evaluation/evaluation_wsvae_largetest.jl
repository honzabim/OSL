using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO

# folderpath = "D:/dev/julia/"
folderpath = "/home/bimjan/dev/julia/"
# folderpath = "D:/dev/"
push!(LOAD_PATH, folderpath, folderpath * "OSL/KNNmemory/")
using KNNmem
using FluxExtensions
using ADatasets
using NearestNeighbors
using StatsBase

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

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

function createSVAEWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE(encoder, decoder, hiddenDim, latentDim, T)
    train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        trainOnLatent!(zfromx(svae, data), collect(labels)) # changes labels to zeros!
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    function justTrain!(data, labels)
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    end

    function learnAnomaly!(data, labels)
        trainOnLatent!(zfromx(svae, data), labels)
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    return svae, learnRepresentation!, learnAnomaly!, classify, justTrain!
end

function rscore(m::SVAE, x)
    xgivenz = m.g(zfromx(m, x))
    return Flux.mse(x, xgivenz)
end

function runExperiment(datasetName, trainall, testall, createModel, anomalyCounts, batchSize = 100, numBatches = 10000)
    anomalyRatios = [0.05, 0.01, 0.005]
    results = []
    for ar in anomalyRatios
        train = ADatasets.subsampleanomalous(trainall, ar)
        test = ADatasets.subsampleanomalous(testall, ar)
        (model, learnRepresentation!, learnAnomaly!, classify, justTrain!) = createModel()
        opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
        cb = Flux.throttle(() -> println("$datasetName AR=$ar : $(justTrain!(train[1], []))"), 5)
        Flux.train!(justTrain!, RandomBatches((train[1], zeros(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)
        # FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2] .- 1), batchSize, numBatches), ()->(), 100)

        learnRepresentation!(train[1], zeros(train[2]))

        rstrn = Flux.Tracker.data(rscore(model, train[1]))
        rstst = Flux.Tracker.data(rscore(model, test[1]))

        balltree = BallTree(Flux.Tracker.data(zfromx(model, train[1])), Euclidean(); reorder = false)
        idxs, dists = knn(balltree, Flux.Tracker.data(zfromx(model, test[1])), 3, false)
        knnscores = map((i, d) -> sum(softmax(1 ./ d)[train[2][i] .== 2]), idxs, dists)
        knnauc = pyauc(test[2] .- 1, knnscores)
        knnroc = roc(test[2] .- 1, map(i -> indmax(counts(train[2][i])) - 1, idxs))
        knnprec = precision(knnroc)
        knnrecall = recall(knnroc)

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
            # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
            auc = pyauc(test[2] .- 1, probScore)
            push!(results, (ac, f1, auc, values, probScore, rstrn, rstst, knnauc, knnprec, knnrecall, ar))
        end
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAElarge/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
datasets = ["breast-cancer-wisconsin", "sonar", "waveform-1"]
difficulties = ["easy", "easy", "easy"]
const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 10000

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for (dn, df) in zip(datasets, difficulties)
    train, test, clusterdness = loadData(dn, df)

    println("$dn")
    println("Running svae...")

    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), 1:5, batchSize, iterations)
    results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [32 64 128 512], [16 32], [1], [0.05])
    results = reshape(results, length(results), 1)
    save(outputFolder * dn * "-svae.jld2", "results", results)
end
