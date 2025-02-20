using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern
using JLD2
using FileIO

# folderpath = "/home/jan/dev/"
folderpath = "D:/dev/julia/"
push!(LOAD_PATH, folderpath * "OSL/KNNmemory/", folderpath)
using FluxExtensions
using ADatasets
using EvalCurves
using Plots
pyplot()

importall KNNmem
include(folderpath * "OSL/SVAE/svae.jl")

const dataPath = "D:/dev/data/loda/public/datasets/numerical"

function augmentModelWithMemory(model, memorySize, keySize, k, labelCount, α = 0.1, T = Float32)
    memory = KNNmemory{T}(memorySize, keySize, k, labelCount, α)
    trainQ!(data, labels) = trainQuery!(memory, model(data), labels)
    trainQOnLatent!(latentData, labels) = trainQuery!(memory, latentData, labels)
    testQ(data) = query(memory, model(data))
    return memory, trainQ!, testQ, trainQOnLatent!
end

function createSVAEWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE(encoder, decoder, hiddenDim, latentDim, T)
    memory, train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        trainOnLatent!(zfromx(svae, data), zeros(collect(labels))) # changes labels to zeros!
        return wloss(svae, data, 0.001, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, 0.01)
    end

    function learnAnomaly!(data, labels)
        trainOnLatent!(zfromx(svae, data), labels)
        return loss(svae, data, 0.005)
    end

    return svae, memory, learnRepresentation!, learnAnomaly!, classify
end

function plotmemory(m::KNNmemory)
    Plots.scatter(m.M[:, 1], m.M[:, 2], zcolor = m.V)
end

normalizecolumns(m) = m ./ sqrt.(sum(m .^ 2, 1) + eps(eltype(Flux.Tracker.data(m))))

# genanomalies(x, y) = gendata(x, y, -1)
# gendata(x, y) = gendata(x, y, 1)
# function gendata(x, y, α)
#     data = rand(x, y) .* 2 .- 1
#     data[1, :] = abs.(data[1, :]) * α
#     data = normalizecolumns(data)
# end
#
# function genset(normal, anomalies)
#     n = gendata(1, normal)
#     a = genanomalies(1, anomalies)
#     train = (hcat(n, a), vcat(zeros(Int64, normal), ones(Int64, anomalies)))
# end

# trainCount = 100
# anomaliesCount = 100
# testCount = 100
# tstAnomalyCount = 5
#
# train = genset(trainCount, anomaliesCount)
# test = genset(testCount, tstAnomalyCount)
#
# Plots.scatter(train[1][1, 1:trainCount], train[1][2, 1:trainCount])
# Plots.scatter!(train[1][1, trainCount + 1:end], train[1][2, trainCount + 1:end])
#
# Plots.scatter(train[1][1, 1:trainCount])
# Plots.scatter!(train[1][1, trainCount + 1:end])

# origlatent = rand(2,1000)
# origlatent[1, 501:1000] = -origlatent[1, 501:1000]
# origlatent[2, 251:500] = -origlatent[2, 251:500]
# origlatent[2, 751:1000] = -origlatent[2, 751:1000]
# origlatent = normalizecolumns(origlatent)
#
# p = Plots.scatter(0,0)
# for i in 1:250:751
#     Plots.scatter!(origlatent[1, i:i + 249], origlatent[2, i:i + 249])
# end
# Plots.display(p)
#
# transofrm = rand(100, 2) .* 2 .- 1

function loadData(datasetName, difficulty)
    train, test, clusterdness = ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.9, "low")
    return ADatasets.subsampleanomalous(train, 0.05), test, clusterdness
end

datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
d = datasets[1]
# for d in datasets
    train, test, clusterdness = loadData(d, "easy")

    idim = size(train[1], 1)
    hdim = 32
    zdim = 2
    numLayers = 3
    nonlinearity = "relu"
    layerType = "Dense"
    memorySize = 32
    k = 5

    svae, mem, learnRepresentation!, learnAnomaly!, classify = createSVAEWithMem(idim, hdim, zdim, numLayers, nonlinearity, layerType, memorySize, k, 1)

    batchSize = 100
    numBatches = 10000

    # Plots.display(plotmemory(mem))

    opt = Flux.Optimise.ADAM(Flux.params(svae), 1e-4)
    FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2]), batchSize, numBatches), ()->(), 100)

    z = Flux.Tracker.data(zfromx(svae, train[1]))
    p = Plots.plot(Plots.scatter(z[1, train[2] .== 1], z[2, train[2] .== 1]), Plots.scatter(z[1, train[2] .== 2], z[2, train[2] .== 2]))
    Plots.title!(d)
    Plots.display(p)
# end

# Plots.display(plotmemory(mem))
#
# learnRepresentation!(train[1], zeros(collect(train[2])))
#
# Plots.display(plotmemory(mem))
#
#
# anomalies = train[1][:, train[2] .== 2] # TODO needs to be shuffled!!!
# anomalyCount = 1:10
# for ac in anomalyCount
#     if ac <= size(anomalies, 2)
#         l = learnAnomaly!(anomalies[:, ac], [2])
#     else
#         break;
#     end
#
#     values, probScore = classify(test[1])
#     values = Flux.Tracker.data(values)
#     probScore = Flux.Tracker.data(probScore)
#
#     rocData = roc(test[2], values)
#     showall(rocData)
#     f1 = f1score(rocData)
#     auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2])...)
#
#     println("AUC: $(auc)")
#     Plots.display(plotmemory(mem))
# end
