using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO

folderpath = "D:/dev/julia/"
# folderpath = "/home/bimjan/dev/julia/"
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

function rscore(m::SVAE, x)
    xgivenz = m.g(zfromx(m, x))
    return Flux.mse(x, xgivenz)
end

function runExperiment(datasetName, trainall, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000, i = 1)
    anomalyRatios = [0.05, 0.01, 0.005]
    results = []
    for ar in anomalyRatios
        println("Running $datasetName with ar: $ar iteration: $i")
        train = ADatasets.subsampleanomalous(trainall, ar)
        (model, learnRepresentation!, learnAnomaly!, classify, justTrain!) = createModel()
        opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
        cb = Flux.throttle(() -> println("$datasetName AR=$ar : $(justTrain!(train[1], []))"), 5)
        Flux.train!(justTrain!, RandomBatches((train[1], zeros(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)
        # FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2] .- 1), batchSize, numBatches), ()->(), 100)
		println("Finished learning $datasetName with ar: $ar iteration: $i")
        mpwmutualinf = meanpairwisemutualinf(Flux.Tracker.data(zfromx(model, train[1])))
        println("Mean pairwise mutualinf: $mpwmutualinf")

        learnRepresentation!(train[1], zeros(train[2]))

        rstrn = Flux.Tracker.data(rscore(model, train[1]))
        rstst = Flux.Tracker.data(rscore(model, test[1]))

        balltree = BallTree(Flux.Tracker.data(zfromx(model, train[1])), Euclidean(); reorder = false)
        idxs, dists = NearestNeighbors.knn(balltree, Flux.Tracker.data(zfromx(model, test[1])), 3, false)
        knnscores = map((i, d) -> sum(softmax(1 ./ d)[train[2][i] .== 2]), idxs, dists)
        knnauc = pyauc(test[2] .- 1, knnscores)
        knnroc = roc(test[2] .- 1, map(i -> indmax(counts(train[2][i])) - 1, idxs))
        knnprec = precision(knnroc)
        knnrecall = recall(knnroc)

        knnanom = kNN.KNNAnomaly(train[1], :gamma)
        knn5auc = pyauc(test[2] .- 1, StatsBase.predict(knnanom, test[1], 5))
        knn9auc = pyauc(test[2] .- 1, StatsBase.predict(knnanom, test[1], 9))
        knn15auc = pyauc(test[2] .- 1, StatsBase.predict(knnanom, test[1], 15))
        knnsqrtauc = pyauc(test[2] .- 1, StatsBase.predict(knnanom, test[1], round(Int, sqrt(size(train[1], 2)))))
        println("knn 5 auc: $knn5auc")
        println("knn 9 auc: $knn9auc")
        println("knn 15 auc: $knn15auc")
        println("knn 15 auc: $knnsqrtauc")

        anomalies = train[1][:, train[2] .- 1 .== 1]
        anomalies = anomalies[:, randperm(size(anomalies, 2))]

        println("set size: $(size(train[1]))")
        println("set size: $(size(hcat(train[1][:, train[2] .== 1], anomalies[:, 1:min(5, size(anomalies, 2))])))")

        knn5anom = kNN.KNNAnomaly(hcat(train[1][:, train[2] .== 1], anomalies[:, 1:min(5, size(anomalies, 2))]) , :gamma)
        knn5a3auc = pyauc(test[2] .- 1, StatsBase.predict(knn5anom, test[1], 3))
        knn5a5auc = pyauc(test[2] .- 1, StatsBase.predict(knn5anom, test[1], 5))
        knn5a9auc = pyauc(test[2] .- 1, StatsBase.predict(knn5anom, test[1], 9))
        knn5asqrtauc = pyauc(test[2] .- 1, StatsBase.predict(knn5anom, test[1], round(Int, sqrt(size(train[1], 2)))))
        println("knn5a 5 auc: $knn5a5auc")
        println("knn5a 9 auc: $knn5a9auc")
        println("knn5a 3 auc: $knn5a3auc")
        println("knn5a 15 auc: $knn5asqrtauc")


        for ac in anomalyCounts
            if ac <= size(anomalies, 2)
                l = learnAnomaly!(anomalies[:, ac], [1])
            else
                println("Not enough anomalies $ac, $(size(anomalies))")
                println("Counts: $(counts(train[2]))")
                break;
            end

            values, probScore = classify(test[1])
            values = Flux.Tracker.data(values)
            probScore = Flux.Tracker.data(probScore)

            rocData = roc(test[2] .- 1, values)
            f1 = f1score(rocData)
            # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
            auc = pyauc(test[2] .- 1, probScore)
            println("mem AUC: $auc")
            # push!(results, (ac, f1, auc, values, probScore, rstrn, rstst, knnauc, knnprec, knnrecall, ar, i, mpwmutualinf, knn5auc, knn9auc, knn15auc, knnsqrtauc, knn5a3auc, knn5a5auc, knn5a9auc, knn5asqrtauc))
        end
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAElargekNN/"
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

for i in 1:10
	for (dn, df) in zip(datasets, difficulties)
	    train, test, clusterdness = loadData(dn, df)

	    println("$dn")
	    println("$(size(train[2]))")
	    println("$(counts(train[2]))")
	    println("Running svae...")

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), 1:5, batchSize, iterations)
	    results = gridSearch(evaluateOneConfig, [32], [3], [3], ["relu"], ["Dense"], [32 128 1024], [16 32], [1], [0.1])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
