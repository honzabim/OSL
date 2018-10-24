using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using ADatasets

# folderpath = "D:/dev/julia/"
folderpath = "/home/bimjan/dev/julia/"
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
    learnAnomaly!(anomaly) = set_anomalous_hypersphere(svae, anomaly)
	learnWithAnomaliesLkh!(data, labels) = wloss_anom_lkh(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))
	learnWithAnomaliesWass!(data, labels) = wloss_anom_wass(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))

    return svae, learnRepresentation!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!
end

function runExperiment(datasetName, trainall, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000, it = 1)
    anomalyRatios = [0.05, 0.01, 0.005]
    results = []
	for method in ["lklh", "wass"]
	    for ar in anomalyRatios
	        println("Running $datasetName with ar: $ar iteration: $it method: $method")
	        train = ADatasets.subsampleanomalous(trainall, ar)
	        (model, learnRepresentation!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!) = createModel()
			learnWithA! = learnWithAnomaliesWass!
			if method == "lklh"
				learnWithA! = learnWithAnomaliesLkh!
			end

			numBatches = 10000

			opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
	        cb = Flux.throttle(() -> println("SVAE $datasetName AR=$ar : $(learnRepresentation!(train[1], []))"), 5)
	        Flux.train!(learnRepresentation!, RandomBatches((train[1], zeros(train[2])), batchSize, numBatches), opt, cb = cb)

	        a_ids = find(train[2] .- 1 .== 1)
	        a_ids = a_ids[randperm(length(a_ids))]

			# Number of batches for learning with anomalies
			numBatches = 5000

	        for ac in anomalyCounts
	            if ac <= length(a_ids)
					if ac == 1
	                	l = learnAnomaly!(train[1][:, a_ids[ac]])
					else
						newlabels = zeros(train[2])
						newlabels[a_ids[1:ac]] .= 1
						opt = Flux.Optimise.ADAM(Flux.params(model), 3e-5)
						cb = Flux.throttle(() -> println("Learning with anomalies: $(learnWithA!(train[1], newlabels))"), 3)
						Flux.train!(learnWithA!, RandomBatches((train[1], newlabels), batchSize, numBatches), opt, cb = cb)
					end
	            else
	                println("Not enough anomalies $ac, $(size(a_ids))")
	                println("Counts: $(counts(train[2]))")
	                break;
	            end

				println("Anomaly HS params are μ: $(model.anom_priorμ) κ: $(model.anom_priorκ)")
	            ascore = Flux.Tracker.data(score(model, test[1]))
	            auc = pyauc(test[2] .- 1, ascore')
	            println("AUC: $auc")

	            push!(results, (method, ac, auc, ascore, ar, it))
	        end
	    end
	end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAElargeSVAEanom/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# datasets = ["breast-cancer-wisconsin", "sonar", "statlog-segment"]
datasets = ["breast-cancer-wisconsin"]
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
	    println("Running svae...")

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAE_anom(size(train[1], 1), p...), 1:10, batchSize, iterations, i)
	    results = gridSearch(evaluateOneConfig, [32], [3], [3], ["relu"], ["Dense"], [0.1])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
