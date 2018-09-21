using Adapt
using Flux, Flux.Data.MNIST
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

function runExperiment(train, test, createModel, batchSize = 100, numBatches = 10000, i = 1)
    results = []
    println("Creating model")
    (model, learnRepresentation!, learnAnomaly!, classify, justTrain!) = createModel()
    println("Initializing optimiser")
    opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
    cb = Flux.throttle(() -> println("$MNIST loss: $(justTrain!(train[1], []))"), 5)
    println("Creating batches")
    batches = RandomBatches((train[1], zeros(train[2]) .+ 10), batchSize, numBatches)
    println("Training")
    Flux.train!(justTrain!, batches, opt, cb = cb)

    learnRepresentation!(train[1], zeros(train[2]) .+ 10)

    perm = randperm(length(train[2]))
    imgs = train[1][:, perm]
    labels = train[2][perm]

    for ic in 10:10:150
        learnAnomaly!(imgs[:, (ic - 9):ic], labels[(ic - 9):ic])

        values, _ = classify(test[1])
        values = Flux.Tracker.data(values)

        er = count(values != test[2]) / length(test[2])

        push!(results, (values, er, ic))
    end

    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAE_MNIST/"
mkpath(outputFolder)


batchSize = 100
iterations = 100

function loadMNIST(set)
    imgs = MNIST.images(set)
    X = hcat(float.(reshape.(imgs, :))...)
    labels = MNIST.labels(set)
    return (X, labels)
end

train = loadMNIST(:train)
test = loadMNIST(:test)

for i in 1:10
    println("$(size(train[2]))")
    println("$(counts(train[2]))")
    println("Running svae on MNIST...")

    evaluateOneConfig = p -> runExperiment(train, test, () -> createSVAEWithMem(size(train[1], 1), p...), batchSize, iterations)
    results = gridSearch(evaluateOneConfig, [32 64 128], [10], [3 4], ["relu"], ["Dense"], [128 256 1024], [16 32 64], [1], [0.1])
    results = reshape(results, length(results), 1)
    save(outputFolder * "MNIST" *  "-$i-svae.jld2", "results", results)
end
