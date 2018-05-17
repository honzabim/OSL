using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern
using JLD

push!(LOAD_PATH, "/home/jan/dev/OSL/KNNmemory", "/home/jan/dev/FluxExtensions.jl/src", "/home/jan/dev/anomaly detection/anomaly_detection/src")
using KNNmem
using FluxExtensions
using AnomalyDetection

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
function createFeedForwardModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, T = Float32)
    model = FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, "", layerType)
    model = Adapt.adapt(T, model)
    train!, classify, _ = augmentModelWithMemory(model, memorySize, latentDim, k, labelCount, α, T)
    return model, train!, train!, classify
end

"""
    createAutoencoderModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, [α = 0.1], [T = Float32])
Creates an autoencoder model that has a kNN memory connected to the latent layer and provides functions for its training and use as a classifier.
"""
function createAutoencoderModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, γ = 0.5, T = Float32)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers, nonlinearity, "", layerType))
    model(x) = decoder(encoder(x))
    train!, classify, trainOnLatent! = augmentModelWithMemory(encoder, memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        latentVariables = encoder(data)
        trainOnLatent!(latentVariables, collect(labels))
        return Flux.mse(decoder(latentVariables), data)
    end

    function learnAnomaly!(data, labels)
        latentVariable = encoder(data)
        return (1 - γ) * Flux.mse(decoder(latentVariable), data) + γ * trainOnLatent!(latentVariable, labels)
    end

    return model, learnRepresentation!, learnAnomaly!, classify
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 1000)
    (model, learnRepresentation!, learnAnomaly!, classify) = createModel()
    opt = Flux.Optimise.ADAM(params(model))
    FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train.data, train.labels), batchSize, numBatches), cbreak = 100)
    results = []
    anomalies = train.data[:, train.labels .== 1] # TODO needs to be shuffled!!!
    for ac in anomalyCounts
        if ac <= size(anomalies, 2)
            l = learnAnomaly!(anomalies[:, ac], 1)
            Flux.Tracker.back!(l)
            opt()
        else
            break;
        end
        values, probs = classify(test.data)
        rocData = roc(test.labels, values)
        push!(results, (ac, f1score(rocData)))
    end
    return results
end

datasets = ["abalone", "breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
difficulties = ["hard", "easy", "easy", "easy", "easy", "medium"]
dataPath = "/home/jan/dev/data/loda/public/datasets/numerical"
allData = AnomalyDetection.loaddata(dataPath)

datasetName = "abalone"
dataset = allData[datasetName]
train, test, clusterdness = AnomalyDetection.makeset(dataset, 0.9, "easy", 0.1, "high")

evaluateOneConfig = p -> runExperiment(datasetName, train, test, () -> createAutoencoderModel(size(train.data, 1), p...), 1:5, 100, 10000)

# inputDim (already in), hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, γ = 0.5
results = gridSearch(evaluateOneConfig, 32, 5, 4, ["relu", "leakyrelu"], ["ResDense"], [128, 256], [32, 64], 1)
