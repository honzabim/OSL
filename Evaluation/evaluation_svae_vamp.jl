using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using UCI
using DataFrames

# folderpath = "D:/dev/julia/"
folderpath = "/home/bimjan/dev/julia/"
# folderpath = "D:/dev/"
push!(LOAD_PATH, folderpath)
using NearestNeighbors
using StatsBase
using InformationMeasures
using Random
using ADatasets

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_vamp.jl")

outputFolder = folderpath * "OSL/experiments/SVAEvamp/"
mkpath(outputFolder)

const dataFolder = outputFolder

"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function createSVAE_vamp(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, pi_count, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_vamp(encoder, decoder, hiddenDim, latentDim, pi_count, T)

    learnRepresentation!(data, foo) = wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
	learnRepresentationWprior!(data, foo) = wlossprior(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    learnAnomaly!(anomaly) = set_anomalous_hypersphere(svae, anomaly)
	learnWithAnomaliesLkh!(data, labels) = wloss_anom_lkh(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))
	learnWithAnomaliesWass!(data, labels) = wloss_anom_wass(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))

    return svae, learnRepresentation!, learnRepresentationWprior!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000, it = 1)
    results = []
	ac = 0

    println("Running $datasetName with iteration: $it")
    (model, learnRepresentation!, learnRepresentationWprior!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!) = createModel()
	learnrep! = learnRepresentation!
	β = model.β

	opt = Flux.Optimise.ADAM(Flux.params(model), 3e-5)
    cb = Flux.throttle(() -> println("VAMP $datasetName it $it : $(learnRepresentation!(train[1], []))"), 5)
    Flux.train!(learnrep!, RandomBatches((train[1], zeros(train[2])), batchSize, numBatches), opt, cb = cb)

	ascore = Flux.Tracker.data(.-pxvita(model, test[1]))
    auc = pyauc(test[2] .- 1, ascore')
	println(size(ascore))
	println(size(test[2]))
    println("AUC vamp: $auc")

	push!(results, (datasetName, ac, auc, ascore, it))

    return results
end

datasets = ["abalone"]
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
	    println("Running vamp...")

		evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAE_vamp(size(train[1], 1), p...), 1:5, batchSize, iterations, i)
		results = gridSearch(evaluateOneConfig, [32], [3], [3], ["relu"], ["Dense"], [0.1 0.5 1. 5], [3 5 10 30 100 500])
		results = reshape(results, length(results), 1)
		save(outputFolder * dn * "-$i-svae.jld2", "results", results)
	end
end
