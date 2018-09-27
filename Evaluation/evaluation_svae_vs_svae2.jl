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
using InformationMeasures
using kNN

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae.jl")
include(folderpath * "OSL/SVAE/svae2.jl")


"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function meanpairwisemutualinf(x)
    dim = size(x, 1)
    mutualinf = 0
    for i in 1:(dim - 1)
        for j in (i + 1):dim
            mutualinf += get_mutual_information(x[i, :], x[j, :], mode = "uniform_count")
        end
    end
    return mutualinf / (dim * (dim - 1) / 2)
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

function createSVAE2WithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae2 = SVAE2(encoder, decoder, hiddenDim, latentDim, T)
    train2!, classify2, trainOnLatent2! = augmentModelWithMemory((x) -> zfromx(svae2, x), memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation2!(data, labels)
        trainOnLatent2!(zfromx(svae2, data), collect(labels)) # changes labels to zeros!
        return wloss(svae2, data, β, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    function justTrain2!(data, labels)
        return wloss(svae2, data, β, (x, y) -> mmd_imq(x, y, 1))
    end

    function learnAnomaly2!(data, labels)
        trainOnLatent2!(zfromx(svae2, data), labels)
        return wloss(svae2, data, β, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    return svae2, learnRepresentation2!, learnAnomaly2!, classify2, justTrain2!
end

function runExperiment(datasetName, trainall, test, createModel, createModel2, anomalyCounts, batchSize = 100, numBatches = 10000, it = 1)
    anomalyRatios = [0.05, 0.01, 0.005]
    results = []
    for ar in anomalyRatios
        println("Running $datasetName with ar: $ar iteration: $it")
        train = ADatasets.subsampleanomalous(trainall, ar)
        (model, learnRepresentation!, learnAnomaly!, classify, justTrain!) = createModel()
		(model2, learnRepresentation2!, learnAnomaly2!, classify2, justTrain2!) = createModel2()

		opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
        cb = Flux.throttle(() -> println("SVAE $datasetName AR=$ar : $(justTrain!(train[1], []))"), 5)
        Flux.train!(justTrain!, RandomBatches((train[1], zeros(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)

		opt2 = Flux.Optimise.ADAM(Flux.params(model2), 1e-4)
        cb = Flux.throttle(() -> println("SVAE2 $datasetName AR=$ar : $(justTrain2!(train[1], []))"), 5)
        Flux.train!(justTrain2!, RandomBatches((train[1], zeros(train[2]) .+ 2), batchSize, numBatches), opt2, cb = cb)

        learnRepresentation!(train[1], zeros(train[2]))
		learnRepresentation2!(train[1], zeros(train[2]))

        anomalies = train[1][:, train[2] .- 1 .== 1]
        anomalies = anomalies[:, randperm(size(anomalies, 2))]

        for ac in anomalyCounts
            if ac <= size(anomalies, 2)
                l = learnAnomaly!(anomalies[:, ac], [1])
				l2 = learnAnomaly2!(anomalies[:, ac], [1])
            else
                println("Not enough anomalies $ac, $(size(anomalies))")
                println("Counts: $(counts(train[2]))")
                break;
            end

            values, probScore = classify(test[1])
            values = Flux.Tracker.data(values)
            probScore = Flux.Tracker.data(probScore)

            auc = pyauc(test[2] .- 1, probScore)
            println("mem1 AUC: $auc")

			values, probScore = classify2(test[1])
			values = Flux.Tracker.data(values)
			probScore = Flux.Tracker.data(probScore)

			auc2 = pyauc(test[2] .- 1, probScore)
			println("mem2 AUC: $auc2")

            push!(results, (ac, auc, auc2, values, probScore, ar, it))
        end
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAElargeVarOfDistances/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# datasets = ["breast-cancer-wisconsin", "sonar", "statlog-segment"]
datasets = ["breast-cancer-wisconsin"]
difficulties = ["easy"]
const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 10000

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for i in 1:30
	for (dn, df) in zip(datasets, difficulties)

	    train, test, clusterdness = loadData(dn, df)

	    println("$dn")
	    println("$(size(train[2]))")
	    println("$(counts(train[2]))")
	    println("Running svae...")

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), () -> createSVAE2WithMem(size(train[1], 1), p...), 1:10, batchSize, iterations, i)
	    results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [128 1024], [16], [1], [0.1])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
