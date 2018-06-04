using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern
using JLD2
using FileIO

push!(LOAD_PATH, "/home/jan/dev/OSL/KNNmemory", "/home/jan/dev/FluxExtensions.jl/src", "/home/jan/dev/anomaly detection/anomaly_detection/src", "/home/jan/dev/EvalCurves.jl/src")
using KNNmem
using FluxExtensions
using AnomalyDetection
using EvalCurves

# TODO export neco, necoJinyho

"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> (p, f(p)), Base.product(parameters...))

"""
    createFeedForwardModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, [α = 0.1], [T = Float32])
Creates a feed-forward model that ends with a kNN memory and provides functions for its training and use as a classifier.
"""
function createFeedForwardModelWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, T = Float32)
    model = FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim, numLayers + 1, nonlinearity, "linear", layerType)
    model = Adapt.adapt(T, model)
    train!, classify, _ = augmentModelWithMemory(model, memorySize, latentDim, k, labelCount, α, T)
    return model, (data, labels) -> train!(data, zeros(collect(labels))), train!, classify
end

function createFeedForwardModel(inputDim, hiddenDim, numberOfLabels, numLayers, nonlinearity, layerType, T = Float32)
    model = FluxExtensions.layerbuilder(inputDim, hiddenDim, numberOfLabels, numLayers + 1, nonlinearity, "linear", layerType)
    push!(model.layers, softmax)
    model = Adapt.adapt(T, model)

    train!(data, labels) = Flux.crossentropy(model(data), Flux.onehotbatch(labels, 0:(numberOfLabels - 1))) + 0.001 * sum(x -> sum(x .^ 2), params(model))

    function classify(data)
        probs = model(data)
        return  Flux.argmax(probs), probs[1, :]
    end
    return model, train!, train!, classify
end

"""
    createAutoencoderModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, [α = 0.1], [γ = 0.5], [T = Float32])
Creates an autoencoder model that has a kNN memory connected to the latent layer and provides functions for its training and use as a classifier.
"""
function createAutoencoderModelWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, γ = 0.5, T = Float32)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))
    model = Chain(encoder, decoder)
    train!, classify, trainOnLatent! = augmentModelWithMemory(encoder, memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        latentVariables = encoder(data)
        trainOnLatent!(latentVariables, zeros(collect(labels))) # changes labels to zeros!
        return Flux.mse(decoder(latentVariables), data)
    end

    function learnAnomaly!(data, labels)
        latentVariable = encoder(data)
        return (1 - γ) * Flux.mse(decoder(latentVariable), data) + γ * trainOnLatent!(latentVariable, labels)
    end

    rscore(data) = Flux.mse(model(data), data)

    return model, learnRepresentation!, learnAnomaly!, classify, rscore
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 1000)
    (model, learnRepresentation!, learnAnomaly!, classify, rscore) = createModel()
    opt = Flux.Optimise.ADAM(params(model))
    FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train.data, train.labels), batchSize, numBatches), cbreak = 1000)
    rstrn = Flux.Tracker.data(rscore(train.data))
    rstst = Flux.Tracker.data(rscore(test.data))
    results = []
    anomalies = train.data[:, train.labels .== 1] # TODO needs to be shuffled!!!
    for ac in anomalyCounts
        if ac <= size(anomalies, 2)
            l = learnAnomaly!(anomalies[:, ac], [1])
            Flux.Tracker.back!(l)
            opt()
        else
            break;
        end

        values, probScore = classify(test.data)
        values = Flux.Tracker.data(values)
        probScore = Flux.Tracker.data(probScore)

        rocData = roc(test.labels, values)
        f1 = f1score(rocData)
        tprvec, fprvec = EvalCurves.roccurve(probScore, test.labels)
        auc = EvalCurves.auc(fprvec, tprvec)
        push!(results, (ac, f1, auc, values, probScore, rstrn, rstst))
    end
    return results
end

outputFolder = "/home/jan/dev/OSL/experiments/findingBestAEWith2LatentConsistency/"
mkpath(outputFolder)

#datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
#difficulties = ["easy", "easy", "easy", "easy", "easy"]

datasets = ["waveform-1"]
difficulties = ["easy"]

dataPath = "/home/jan/dev/data/loda/public/datasets/numerical"
allData = AnomalyDetection.loaddata(dataPath)

batchSize = 100
iterations = 10000

loadData(datasetName, difficulty) = AnomalyDetection.makeset(allData[datasetName], 0.9, difficulty, 0.1, "high")

for i in 1:100
for (dn, df) in zip(datasets, difficulties)
    train, test, clusterdness = loadData(dn, df)

    println("$dn")

    iterations = 10000
    println("Running autoencoder...")

    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createAutoencoderModelWithMem(size(train.data, 1), p...), 1:5, batchSize, iterations)
    results = gridSearch(evaluateOneConfig, [16], [2], [4], ["leakyrelu"], ["Dense"], [1024], [32], 1)
    #println(results)
    results = reshape(results, length(results), 1)
    #println(typeof(results))
    save(outputFolder * dn * ".$i-autoencoder.jld2", "results", results)
    #
    # println("Running ff with memory...")
    #
    # evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createFeedForwardModelWithMem(size(train.data, 1), p...), 1:5, batchSize, iterations)
    # results = gridSearch(evaluateOneConfig, [8 16 32], [4 16], [3 4 5], ["leakyrelu"], ["Dense", "ResDense"], [1024], [64], 2)
    # #println(results)
    # results = reshape(results, length(results), 1)
    # #println(typeof(results))
    # save(outputFolder * dn * "-ffMem.jld2", "results", results)

    # iterations = 100000
    # println("Running ff...")
    #
    # evaluateOneConfig = p -> (println(p); runExperiment(dn, train, test, () -> createFeedForwardModel(size(train.data, 1), p...), 1:5, batchSize, iterations))
    # results = gridSearch(evaluateOneConfig, [8 16 32], 2, [3 4 5], ["leakyrelu"], ["Dense", "ResDense"])
    # #println(results)
    # results = reshape(results, length(results), 1)
    # #println(typeof(results))
    # save(outputFolder * dn * "-ff.jld2", "results", results)
end
end
