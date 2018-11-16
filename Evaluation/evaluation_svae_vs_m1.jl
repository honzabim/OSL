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
using HypothesisTests

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_anom.jl")
include(folderpath * "OSL/SVAE/vae.jl")

outputFolder = folderpath * "OSL/experiments/SVAEvsM1pval/"
mkpath(outputFolder)

const dataFolder = outputFolder

loadExperiment(filePath) = load(filePath)["results"]

params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :β, :method, :anomaliesSeen, :auc, :i, :model, :dataset]
types = [Int, Int, Int, String, String, Float64, String, Int, Float64, Int, String, String]

function processFile!(dataframe, model, dataset)
    for i in 1:100
        println("Processing $model $dataset $i")
        filename = dataFolder * dataset * "-$i-" * model * ".jld2"
        if !isfile(filename)
            println("$filename not found.")
        else
            results = loadExperiment(filename)
            for i in 1:length(results)
                for j in 1:length(results[1][2])
                    pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:6]...) : vcat(results[i][1]..., -1, -1)
                    # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                    push!(dataframe, vcat(pars..., results[i][2][j][1:3]..., results[i][2][j][5]..., model, dataset))
                end
            end
        end
    end
end


"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_anom(encoder, decoder, hiddenDim, latentDim, β, T)

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
	for method in ["wass"] # ["lklh", "wass"]
        println("Running $datasetName with iteration: $it method: $method")
        (model, learnRepresentation!, learnRepresentationWprior!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!) = createModel()
		learnrep! = learnRepresentation!
		if (method == "wassprior")
			learnrep! = learnRepresentationWprior!
		end
		β = model.β

		opt = Flux.Optimise.ADAM(Flux.params(model), 3e-5)
        cb = Flux.throttle(() -> println("SVAE $datasetName it $it : $(learnRepresentation!(train[1], []))"), 5)
        Flux.train!(learnrep!, RandomBatches((train[1], zeros(train[2])), batchSize, numBatches), opt, cb = cb)

		ascore = Flux.Tracker.data(.-pxvita(model, test[1]))
        auc = pyauc(test[2], ascore')
		println(size(ascore))
		println(size(test[2]))
        println("AUC svae: $auc")

		push!(results, (method, ac, auc, ascore, it))

		if (method == "wassprior")
			ascore = Flux.Tracker.data(pz(model, test[1]))
	        auc = pyauc(test[2], ascore')
			println(size(ascore))
			println(size(test[2]))
	        println("AUC svae pz: $auc")

			push!(results, (method * " pz", ac, auc, ascore, it))
		end

		if method == "wass"
			method = "m1"

			encoder = Adapt.adapt(Float64, FluxExtensions.layerbuilder(size(train[1], 1), 32, 4, 3, "relu", "", "Dense"))
		    decoder = Adapt.adapt(Float64, FluxExtensions.layerbuilder(2, 32, size(train[1], 1), 3, "relu", "linear", "Dense"))

			model = VAE(encoder, decoder, β, :unit)
			opt = Flux.Optimise.ADAM(Flux.params(model), 3e-5)
	        cb = Flux.throttle(() -> println("M1 $datasetName it $it : $(loss(model, train[1]))"), 5)
	        Flux.train!((x, y) -> loss(model, x), RandomBatches((train[1], zeros(train[2])), batchSize, numBatches), opt, cb = cb)

			ascore = Flux.Tracker.data(.-pxvita(model, test[1]))
            auc = pyauc(test[2], ascore')
			println(size(ascore))
			println(size(test[2]))
            println("AUC m1: $auc")

			push!(results, (method, ac, auc, ascore, it))

		end

		#		learnWithA! = learnWithAnomaliesWass!
		# if method == "lklh"
		# 	learnWithA! = learnWithAnomaliesLkh!
		# end

        # a_ids = find(train[2] .== 1)
        # a_ids = a_ids[randperm(length(a_ids))]
		#
		# # Number of batches for learning with anomalies
		# numBatches = 5000
		#
        # for ac in anomalyCounts
        #     if ac <= length(a_ids)
		# 		if ac == 1
        #         	l = learnAnomaly!(train[1][:, a_ids[ac]])
		# 		else
		# 			newlabels = zeros(train[2])
		# 			newlabels[a_ids[1:ac]] .= 1
		# 			opt = Flux.Optimise.ADAM(Flux.params(model), 1e-5)
		# 			cb = Flux.throttle(() -> println("Learning with anomalies: $(learnWithA!(train[1], newlabels))"), 3)
		# 			Flux.train!(learnWithA!, RandomBatches((train[1], newlabels), batchSize, numBatches), opt, cb = cb)
		# 		end
        #     else
        #         println("Not enough anomalies $ac, $(size(a_ids))")
        #         println("Counts: $(counts(train[2]))")
        #         break;
        #     end
		#
		# 	println("Anomaly HS params are μ: $(model.anom_priorμ) κ: $(model.anom_priorκ)")
        #     ascore = Flux.Tracker.data(score(model, test[1]))
        #     auc = pyauc(test[2] .- 1, ascore')
        #     println("AUC: $auc")
		#
        #     push!(results, (method, ac, auc, ascore, ar, it))
        # end
	end
    return results
end

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# datasets = ["breast-cancer-wisconsin", "sonar", "statlog-segment"]
dataset = "breast-cancer-wisconsin"
# dataset = "ecoli"
# dataset = "abalone"
batchSize = 100
iterations = 100

if length(ARGS) != 0
    dataset = ARGS[1]
end

data, normal_labels, anomaly_labels = UCI.get_umap_data(dataset)

subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
for (subdata, class_label) in subdatasets
	for i in 1:100
		println(dataset*" "*class_label * " it: $i")

		_X_tr, _y_tr, _X_tst, _y_tst = UCI.split_data(subdata, 0.8)
		_X_tr .-= minimum(_X_tr)
		_X_tr ./= maximum(_X_tr)
		_X_tst .-= minimum(_X_tst)
		_X_tst ./= maximum(_X_tst)
		train = (_X_tr, _y_tr)
		test = (_X_tst, _y_tst)

		evaluateOneConfig = p -> runExperiment(dataset, train, test, () -> createSVAE_anom(size(train[1], 1), p...), 1:5, batchSize, iterations, i)
		results = gridSearch(evaluateOneConfig, [32], [3], [3], ["relu"], ["Dense"], [1e-3 0.01 0.1 0.5 1. 5])
		results = reshape(results, length(results), 1)
		save(outputFolder * dataset*" "*class_label *  "-$i-svae.jld2", "results", results)

		allData = DataFrame(types, params, 0)
		processFile!(allData, "svae", dataset * " " * class_label)

		iters = maximum(allData[:i])
		maxed = []

		for i in 1:iters
			maxm1 = maximum(allData[(allData[:method] .== "m1") .& (allData[:i] .== i), :][:auc])
			maxwass = maximum(allData[(allData[:method] .== "wass") .& (allData[:i] .== i), :][:auc])
			push!(maxed, [maxm1, maxwass])
		end
		maxed = hcat(maxed...)

		pval = pvalue(SignedRankTest(maxed[1, :], maxed[2, :]))
		println(dataset*" "*class_label * " it: $i pval = $pval")

		if (i > 10) & (pval < 0.01)
			break
		end
	end
end
