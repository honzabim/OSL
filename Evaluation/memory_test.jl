using Adapt
using Flux
using MLBase: roc, f1score
using MLDataPattern
using JLD2
using FileIO

folderpath = "/home/jan/dev/"
# folderpath = "D:/dev/"
push!(LOAD_PATH, folderpath * "OSL/KNNmemory", folderpath * "FluxExtensions.jl/src", folderpath * "anomaly detection/anomaly_detection/src", folderpath * "EvalCurves.jl/src", folderpath)
using FluxExtensions
using AnomalyDetection
using EvalCurves
using ADatasets
using Plots
plotly()
importall KNNmem

include(folderpath * "OSL/SVAE/svae.jl")

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
        return loss(svae, data)
    end

    function learnAnomaly!(data, labels)
        trainOnLatent!(zfromx(svae, data), labels)
        return loss(svae, data)
    end

    return svae, memory, learnRepresentation!, learnAnomaly!, classify
end

normalizecolumns(m) = m ./ sqrt.(sum(m .^ 2, 1) + eps(eltype(Flux.Tracker.data(m))))

idim = 5
hdim = 8
zdim = 2
numLayers = 3
nonlinearity = "leakyrelu"
layerType = "Dense"
memorySize = 32
k = 5

svae, mem, learnRepresentation!, learnAnomaly!, classify = createSVAEWithMem(idim, hdim, zdim, numLayers, nonlinearity, layerType, memorySize, k, 1)

genanomalies(x, y) = gendata(x, y, -1)
gendata(x, y) = gendata(x, y, 1)
function gendata(x, y, α)
    data = rand(x, y) .* 2 .- 1
    data[1, :] = abs.(data[1, :]) * α
    data = normalizecolumns(data)
end

function genset(normal, anomalies)
    n = gendata(1, normal)
    a = genanomalies(1, anomalies)
    train = (hcat(n, a), vcat(zeros(Int64, normal), ones(Int64, anomalies)))
end

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

origlatent = rand(2,1000)
origlatent[1, 501:1000] = -origlatent[1, 501:1000]
origlatent[2, 251:500] = -origlatent[2, 251:500]
origlatent[2, 751:1000] = -origlatent[2, 751:1000]
origlatent = normalizecolumns(origlatent)

p = Plots.scatter(0,0)
for i in 1:250:751
    Plots.scatter!(origlatent[1, i:i+249], origlatent[2, i:i+249])
end
Plots.display(p)

transofrm = rand(5, 2) .* 2 .- 1

trdata = transofrm * origlatent
trlabels = vcat(repmat([1], 250, 1), repmat([2], 250, 1), repmat([3], 250, 1), repmat([4], 250, 1))
train = (trdata, vec(trlabels))

batchSize = 100
numBatches = 5000

opt = Flux.Optimise.ADAM(Flux.params(svae))
FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2]), batchSize, numBatches), ()->(), 100)

z = Flux.Tracker.data(zfromx(svae, train[1]))
# Plots.scatter(z[1, 1:trainCount], z[2, 1:trainCount])
# Plots.scatter!(z[1, trainCount + 1:end], z[2, trainCount + 1:end])
p = Plots.scatter(0,0)
for i in 1:250:751
    Plots.scatter!(z[1, i:i+249], z[2, i:i+249])
end
Plots.display(p)

xgivenz = Flux.Tracker.data(infer(svae, train[1]))
Plots.scatter(xgivenz[1, 1:trainCount], xgivenz[2, 1:trainCount])
Plots.scatter!(xgivenz[1, trainCount + 1:end], xgivenz[2, trainCount + 1:end])

anomalies = train[1][:, train[2] .== 1] # TODO needs to be shuffled!!!
anomalyCount = 1:5
for ac in anomalyCount
    if ac <= size(anomalies, 2)
        l = learnAnomaly!(anomalies[:, ac], [1])
    else
        break;
    end

    values, probScore = classify(test[1])
    values = Flux.Tracker.data(values)
    probScore = Flux.Tracker.data(probScore)

    rocData = roc(test[2], values)
    showall(rocData)
    f1 = f1score(rocData)
    auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2])...)
end
