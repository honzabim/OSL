using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern
# using JLD2
# using FileIO

push!(LOAD_PATH, "/home/jan/dev/OSL/KNNmemory", "/home/jan/dev/FluxExtensions.jl/src", "/home/jan/dev/anomaly detection/anomaly_detection/src", "/home/jan/dev/EvalCurves.jl/src", "/home/jan/dev/ADatasets.jl/src")
using KNNmem
using FluxExtensions
using ADatasets
using AnomalyDetection

# TODO export neco, necoJinyho

"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> (p, f(p)), Base.product(parameters...))

"""
    createAutoencoderModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, [α = 0.1], [γ = 0.5], [T = Float32])
Creates an autoencoder model that has a kNN memory connected to the latent layer and provides functions for its training and use as a classifier.
"""
function createAutoencoderModelWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, γ = 0.5, T = Float32)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))
    model = Chain(encoder, decoder)
    # train!, classify, trainOnLatent! = augmentModelWithMemory(encoder, memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        latentVariables = encoder(data)
        # trainOnLatent!(Flux.data(latentVariables), zeros(collect(labels))) # changes labels to zeros!
        return Flux.mse(decoder(latentVariables), data)
    end

    function learnAnomaly!(data, labels)
        latentVariable = encoder(data)
        return (1 - γ) * Flux.mse(decoder(latentVariable), data) + γ * trainOnLatent!(latentVariable, labels)
    end

    return model, learnRepresentation!, learnAnomaly!, nothing
    # return model, learnRepresentation!, learnAnomaly!, classify
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 1000)
    (model, learnRepresentation!, learnAnomaly!, classify) = createModel()
    opt = Flux.Optimise.ADAM(params(model), 0.000001)
    FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train.data, train.labels), batchSize, numBatches), cbreak = 1000)
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
        push!(results, (ac, f1, auc, values, probScore))
    end
    return results
end

outputFolder = "/home/jan/dev/OSL/experiments/first/"
mkpath(outputFolder)

datasets = ["abalone", "breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1", "yeast"]
difficulties = ["hard", "easy", "easy", "easy", "easy", "medium"]

#datasets = ["yeast"]
#difficulties = ["medium"]

idir = "/home/jan/dev/data/loda/public/datasets/numerical"
batchSize = 100
iterations = 100000

train, test, clusterdness = ADatasets.makeset(ADatasets.loaddataset("abalone","easy",idir)..., 0.9, 0.1, "high")
model, learnRepresentation!, learnAnomaly!, classify = createAutoencoderModelWithMem(size(train[1], 1), 16,4,3,"leakyrelu","Dense",1024,64,1)
opt = Flux.Optimise.ADAM(params(model))
loss(data, labels) = Flux.mse(model(data), data)
FluxExtensions.learn(loss, opt, RandomBatches(train, 100, 10000), cbreak = 1000)
FluxExtensions.learn(learnRepresentation!, opt, RandomBatches(train, 100, 10000), cbreak = 1000)
