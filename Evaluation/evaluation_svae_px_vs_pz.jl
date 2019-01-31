using Adapt
using Flux
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using ADatasets
using StatsBase
using Random
using LinearAlgebra
using EvalCurves

folderpath = "D:/dev/julia/"
# folderpath = "/home/bimjan/dev/julia/"

include(folderpath * "OSL/SVAE/svae.jl")

outputFolder = folderpath * "OSL/experiments/WSVAE_PXV_vs_PZ_KLD/"
mkpath(outputFolder)

datasets = ["cardiotocography"]
difficulties = ["easy"]

if length(ARGS) != 0
    datasets = [ARGS[1]]
end

const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 10000



"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function createSVAE(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE(encoder, decoder, hiddenDim, latentDim, T)
    train! = (data, labels) -> loss(svae, data, β)

    return svae, train!
end

computeauc(score, labels) = EvalCurves.auc(EvalCurves.roccurve(score, labels)...)

function runExperiment(datasetName, trainall, test, createModel, batchSize, numBatches, it)
    anomalyRatios = [0.05 0.01 0.005]
    results = []
    for ar in anomalyRatios
        println("Running $datasetName with ar: $ar iteration: $it")
		train = ADatasets.subsampleanomalous(trainall, ar)
        (model, train!) = createModel()
        opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
        cb = Flux.throttle(() -> println("$datasetName AR=$ar : $(train!(train[1], []))"), 5)
        Flux.train!(train!, RandomBatches((train[1], train[2]), batchSize, numBatches), opt, cb = cb)
		println("Finished learning $datasetName with ar: $ar iteration: $it")

		pxv = vec(collect(.-pxvita(model, test[1])'))
		pzs = vec(collect(.-pz(model, test[1])'))
		println(size(pxv))
		auc_pxv = computeauc(pxv, test[2] .- 1)
		auc_pz = computeauc(pzs, test[2] .- 1)
		println("P(X) Vita AUC  $auc_pxv on $datasetName with ar: $ar iteration: $it")
		push!(results, (auc_pxv, auc_pz, ar, it))
    end
    return results
end

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

for i in 1:10
	for (dn, df) in zip(datasets, difficulties)

	    train, test, clusterdness = loadData(dn, df)

	    println("$dn")
	    println("Running svae...")

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAE(size(train[1], 1), p...), batchSize, iterations, i)
	    results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [0.01 .1 1 10])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
