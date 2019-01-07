using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using ADatasets

folderpath = "D:/dev/julia/"
# folderpath = "/home/bimjan/dev/julia/"
# folderpath = "D:/dev/"
push!(LOAD_PATH, folderpath)
using NearestNeighbors
using StatsBase
using InformationMeasures

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_anom.jl")


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

function createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_anom(encoder, decoder, hiddenDim, latentDim, T)

	learnRepresentation!(data, foo) = wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
	learnPrintingRepresentation!(data, foo) = printingwloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    learnAnomaly!(μ) = set_anomalous_μ(svae, μ)
	learnWithAnomaliesWass!(data, labels, a) = wloss_anom_vasek(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1), a)
	learnWithAnomaliesPrintingWass!(data, labels) = printing_wloss_anom_vasek(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))

    return svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesWass!, learnWithAnomaliesPrintingWass!
end

function runExperiment(datasetName, trainall, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000, it = 1)
    anomalyRatios = [0.05, 0.01, 0.005]
    results = []
    for ar in anomalyRatios
		for α in [0.125 0.25 0.5 0.75 0.875]
	        println("Running $datasetName with ar: $ar iteration: $it α: $α")
	        train = ADatasets.subsampleanomalous(trainall, ar)
	        (model, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesWass!, learnWithAnomaliesPrintingWass!) = createModel()
			numBatches = 10000

			opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
	        cb = Flux.throttle(() -> println("SVAE $datasetName AR=$ar : $(learnRepresentation!(train[1], []))"), 5)
	        Flux.train!(learnRepresentation!, RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

			pxv = collect(.-pxvita(model, test[1])')
			auc_pxv = pyauc(test[2] .- 1, pxv)
			println("P(X) Vita AUC = $auc_pxv on $datasetName with ar: $ar iteration: $it")

	        a_ids = findall(train[2] .- 1 .== 1)
	        a_ids = a_ids[randperm(length(a_ids))]

			z = Flux.Tracker.data(zparams(model, train[1])[1])
			μnormal = normalize(vec(mean(z, dims = 2)))
			learnAnomaly!(.-μnormal)

			# Number of batches for learning with anomalies
			numBatches = 3000

	        for ac in anomalyCounts
	            if ac <= length(a_ids)
					newlabels = zero(train[2])
					newlabels[a_ids[1:ac]] .= 1
					opt = Flux.Optimise.ADAM(Flux.params(model), 1e-5)
					cb = Flux.throttle(() -> println("Learning with anomalies: $(((d, l) -> learnWithAnomaliesWass!(d, l, α))(train[1], newlabels))"), 3)
					Flux.train!((d, l) -> learnWithAnomaliesWass!(d, l, α), RandomBatches((train[1], newlabels), batchSize, numBatches), opt, cb = cb)
				else
	                println("Not enough anomalies $ac, $(size(a_ids))")
	                println("Counts: $(counts(train[2]))")
	                break;
	            end
				println("Anomaly HS params are μ: $(model.anom_priorμ) κ: $(model.anom_priorκ)")
	            ascore = collect(Flux.Tracker.data(score(model, test[1]))')
	            auc = pyauc(test[2] .- 1, ascore)
	            println("AUC: $auc")

	            push!(results, (ac, auc, auc_pxv, ascore, ar, it, α))
	        end
		end
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAElongSVAEanom/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# datasets = ["breast-cancer-wisconsin", "sonar", "statlog-segment"]
datasets = ["breast-cancer-wisconsin"]
difficulties = ["easy"]
const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 100

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
	    println("Running svae...")

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAE_anom(size(train[1], 1), p...), 1:5, batchSize, iterations, i)
	    results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [0.01 0.1 1 10 100])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
