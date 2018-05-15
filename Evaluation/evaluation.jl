using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern

push!(LOAD_PATH, "/home/jan/dev/OSL/KNNmemory", "/home/jan/dev/FluxExtensions.jl/src", "/home/jan/dev/anomaly detection/anomaly_detection/src")
using KNNmem
using FluxExtensions
using AnomalyDetection

# TODO export neco, necoJinyho

function gridSearch(f, parameters...)
    errs = map(p -> (p, f(p)), Base.product(parameters...))
end

function createFeedForwardModel(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, T = Float32)
    model = FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, "", layerType)
    model = Adapt.adapt(T, model)
    train!, classify, _ = augmentModelWithMemory(model, memorySize, latentDim, k, labelCount, α, T)
    return model, train!, train!, classify
end

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

function runExperiment(train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 1000)
    (model, learnRepresentation!, learnAnomaly!, classify) = createModel()
    opt = Flux.Optimise.ADAM(params(model))
    FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train.data, train.labels), batchSize, numBatches), cbreak = 100)
    results = []
    anomalies = train.data[:, train.labels .== 1] # TODO needs to be shuffled!!!
    for ac in anomalyCounts
        if ac <= size(anomalies, 2)
            learnAnomaly!(anomalies[:, ac], 1)
        else
            break;
        end
        values, _ = classify(test.data)
        rocData = roc(test.labels, values)
        push!(results, (ac, f1score(rocData)))
    end
    return results
end

dataPath = "/home/jan/dev/data/loda/public/datasets/numerical"
allData = AnomalyDetection.loaddata(dataPath)
dataset = allData["abalone"]
train, test, clusterdness = AnomalyDetection.makeset(dataset, 0.9, "easy", 0.05, "high")

evaluateOneConfig = p -> runExperiment(train, test, () -> createAutoencoderModel(size(train.data, 1), p...), 1:5, 100, 10000)

# inputDim (already in), hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, γ = 0.5
results = gridSearch(evaluateOneConfig, 32, 5, [3, 4, 5], ["relu", "leakyrelu"], ["Dense", "ResDense"], [128, 256, 512, 1024], [32, 64], 2)
