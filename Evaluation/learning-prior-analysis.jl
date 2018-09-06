using Adapt
using Flux
using MLBase: roc, f1score, precision, recall, true_positive, false_positive, true_negative, false_negative
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
using Plots
using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
	return pyauc, pyfpr, pytpr
end

include(folderpath * "OSL/SVAE/svae2.jl")

function augmentModelWithMemory(model, memorySize, keySize, k, labelCount, α = 0.1, T = Float32)
    memory = KNNmemory{T}(memorySize, keySize, k, labelCount, α)
    trainQ!(data, labels) = trainQuery!(memory, model(data), labels)
    trainQOnLatent!(latentData, labels) = trainQuery!(memory, latentData, labels)
    testQ(data) = KNNmem.query(memory, model(data))
    return memory, trainQ!, testQ, trainQOnLatent!
end

function createSVAEWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE(encoder, decoder, hiddenDim, latentDim, T)
    mem, train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        trainOnLatent!(zfromx(svae, data), collect(labels)) # changes labels to zeros!
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    function justTrain!(data, labels)
        return  wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    end

    function learnAnomaly!(data, labels)
        trainOnLatent!(zfromx(svae, data), labels)
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    return svae, mem, learnRepresentation!, learnAnomaly!, classify, justTrain!
end

function plotmemory(m::KNNmemory)
    p = Plots.scatter3d(m.M[:, 1], m.M[:, 2], m.M[:, 3], zcolor = m.V)
end

const dataPath = folderpath * "data/loda/public/datasets/numerical"
loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

d = "breast-cancer-wisconsin"
trainall, test, clusterdness = loadData(d, "easy")

idim = size(trainall[1], 1)
hdim = 32
zdim = 3
numLayers = 4
nonlinearity = "relu"
layerType = "Dense"
memorySize = 128
k = 32

svae, mem, learnRepresentation!, learnAnomaly!, classify, justTrain! = createSVAEWithMem(idim, hdim, zdim, numLayers, nonlinearity, layerType, memorySize, k, 1, 0.1)

batchSize = 100
numBatches = 8000
ar = 0.05
train = ADatasets.subsampleanomalous(trainall, ar)

plotly()

function rscore(m::SVAE, x)
    xgivenz = m.g(zfromx(m, x))
    return Flux.mse(x, xgivenz)
end

# Plots.display(plotmemory(mem))

opt = Flux.Optimise.ADAM(Flux.params(svae), 1e-4)
cb = Flux.throttle(() -> println("$d AR=$ar : $(justTrain!(train[1], []))"), 5)
Flux.train!(justTrain!, RandomBatches((train[1], zeros(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)

println("prior μ is: $(svae.priorμ) κ is: $(svae.priorκ)")

z = Flux.Tracker.data(zfromx(svae, train[1]))
p1 = Plots.scatter3d(z[1, train[2] .== 1], z[2, train[2] .== 1], z[3, train[2] .== 1])
p2 = Plots.scatter3d!(z[1, train[2] .== 2], z[2, train[2] .== 2], z[3, train[2] .== 2])
μend = 1.3 * svae.priorμ
p2 = Plots.plot!([0, μend[1]], [0, μend[2]], [0, μend[3]], color = "red", linewidth = "3")
Plots.title!("Train")
Plots.display(p2)

# z = Flux.Tracker.data(zfromx(svae, test[1]))
# p1 = Plots.scatter3d(z[1, test[2] .== 1], z[2, test[2] .== 1], z[3, test[2] .== 1])
# p2 = Plots.scatter3d!(z[1, test[2] .== 2], z[2, test[2] .== 2], z[3, test[2] .== 2])
# Plots.title!("Test")
# Plots.display(p2)

# balltree = BallTree(Flux.Tracker.data(zfromx(svae, train[1])), Euclidean(); reorder = false)
# idxs, dists = knn(balltree, Flux.Tracker.data(zfromx(svae, test[1])), 1, false)
# knnscores = map((i, d) -> sum(softmax(1 ./ d)[train[2][i] .== 2]), idxs, dists)
# knnauc, _, _ = pyauc(test[2] .- 1, knnscores)
# knnroc = roc(test[2] .- 1, map(i -> indmax(counts(train[2][i], 2)) - 1, idxs))
# knnprec = precision(knnroc)
# knnrecall = recall(knnroc)
# println("kNN tp: $(true_positive(knnroc)) fp: $(false_positive(knnroc)) tn: $(true_negative(knnroc)) fn: $(false_negative(knnroc))")
#
# println("kNN AUC: $knnauc")
learnRepresentation!(train[1], zeros(collect(train[2])))

anomalies = train[1][:, train[2] .- 1 .== 1]
anomalies = anomalies[:, randperm(size(anomalies, 2))]

println("rscore of train: $(rscore(svae, train[1]))")
println("rscore of anomalies: $(rscore(svae, anomalies))")

anomalyCounts = 1:5
# aucp = Plots.plot()
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
    auc, fp, tp = pyauc(test[2] .- 1, probScore)
	# Plots.plot!(aucp, fp, tp)
    println("mem AUC: $auc")
	println("mem tp: $(true_positive(rocData)) fp: $(false_positive(rocData)) tn: $(true_negative(rocData)) fn: $(false_negative(rocData))")
    # Plots.display(plotmemory(mem))
    # Plots.title!("After $ac anomalies")
end

# Plots.display(aucp)
