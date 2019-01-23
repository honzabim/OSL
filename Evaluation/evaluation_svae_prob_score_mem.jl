using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
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
using StatsBase
using Random

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_mem.jl")


"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function createSVAEWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, β, loss_α, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE(encoder, decoder, hiddenDim, latentDim, T)
	# mem, train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, memorySize, labelCount, α, T) # TODO the second memsize should be k !!!
    mem, train!, classify, trainOnLatent!, classifyOnLatent = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, inputDim, memorySize, labelCount, x -> zparams(svae, x)[1], α, T) # TODO the second memsize should be k !!!

    function learnRepresentation!(data, labels)
        trainOnLatent!(data, collect(labels)) # changes labels to zeros!
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    end

    function justTrain!(data, labels)
        return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    end

	# function learnWithAnomalies!(data, labels, anoms)
    function learnWithAnomalies!(data, labels)
        #trainOnLatent!(data, labels)
        # return mem_wloss(svae, mem, hcat(data, anoms[:, 1]), vcat(labels, 1), β, (x, y) -> mmd_imq(x, y, 1), loss_α)
		return mem_wloss(svae, mem, data, labels, β, (x, y) -> mmd_imq(x, y, 1), loss_α)
    end

    return mem, svae, learnRepresentation!, learnWithAnomalies!, classify, justTrain!, classifyOnLatent
end

log_normal(x) = - sum((x .^ 2), dims = 1) ./ 2 .- size(x, 1) .* log(2π) ./ 2
log_normal(x, μ) = log_normal(x - μ)

# Likelihood estimation of a sample x under VMF with given parameters taken from https://pdfs.semanticscholar.org/2b5b/724fb175f592c1ff919cc61499adb26996b1.pdf
# normalizing constant for density function of VMF
c(p, κ) = κ ^ (p / 2 - 1) / ((2π) ^ (p / 2) * besseli(p / 2 - 1, κ))

# log likelihood of one sample under the VMF dist with given parameters
log_vmf_c(x, μ, κ) = κ * μ' * x # .+ log(c(length(μ), κ)) in this case we don't need this as it is in both nominator and denominator in the fraction

function pxvita(m::SVAE, x)
	μz, κz = zparams(m, x)
	xgivenz = m.g(μz)
	Flux.Tracker.data(log_normal(xgivenz, x))
end

function f3score(memory::KNNmemory, model::SVAE, z::Vector, x::Vector, κ)
	prob = exp.(log_normal(Flux.Tracker.data(model.g(zparams(model, memory.M')[1])), repeat(x, 1, size(memory.M, 1))) .+ log_vmf_c(Flux.Tracker.data(zparams(model, memory.M')[1]), z, κ))
	sumanom = sum(prob[memory.V .== 1])
	sumnormal = sum(prob[memory.V .== 0])
	return sumanom / (sumnormal + sumanom)
end

function f3score(memory::KNNmemory, model::SVAE, x, κ)
	(μz, _) = zparams(model, x)
	μz = Flux.Tracker.data(μz)
	score = []
	for i in 1:size(x, 2)
		push!(score, f3score(memory, model, μz[:, i], Flux.Tracker.data(x[:, i]), κ))
	end
	return score
end

function runExperiment(datasetName, trainall, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000, it = 1)
	# anomalyRatios = [0.05, 0.01, 0.005]
	anomalyRatios = [0.01]
    results = []
    for ar in anomalyRatios
        println("Running $datasetName with ar: $ar iteration: $it")
        train = ADatasets.subsampleanomalous(trainall, ar)
        (mem, model, learnRepresentation!, learnWithAnomalies!, classify, justTrain!, classifyOnLatent) = createModel()

		numBatches = 10000
        opt = Flux.Optimise.ADAM(Flux.params(model), 1e-4)
        cb = Flux.throttle(() -> println("$datasetName AR=$ar : $(justTrain!(train[1], []))"), 5)
        Flux.train!(justTrain!, RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)
		println("Finished learning $datasetName with ar: $ar iteration: $it")

        learnRepresentation!(train[1], zero(train[2]))
		println("Added all normal to memory")

		pxv = collect(.-pxvita(model, test[1])')
		auc_pxv = pyauc(test[2] .- 1, pxv)
		println("P(X) Vita AUC = $auc_pxv on $datasetName with ar: $ar iteration: $it")

		a_ids = findall(train[2] .- 1 .== 1)
		a_ids = a_ids[randperm(length(a_ids))]

		numBatches = 4000

        for ac in anomalyCounts
			println("Anomaly count $ac")
            if ac <= length(a_ids)
                l = learnRepresentation!(train[1][:, a_ids[ac]], [1])
				newlabels = zero(train[2])
				newlabels[a_ids[1:ac]] .= 1
				# anoms = train[1][:, a_ids[rand(1:ac, length(newlabels))]]
				opt = Flux.Optimise.ADAM(Flux.params(model), 3e-5)
		        cb = Flux.throttle(() -> println("$datasetName AR=$ar : $(justTrain!(train[1], newlabels))"), 5)
				println("Starting learning")
		        Flux.train!(learnWithAnomalies!, RandomBatches((train[1], newlabels), batchSize, numBatches), opt, cb = cb)
            else
                println("Not enough anomalies $ac, $(size(anomalies))")
                println("Counts: $(counts(train[2]))")
                break;
            end

			for κ in [0.5, 1, 3, 5, 10, 50]
	            values, probScore = classifyOnLatent(test[1], κ)
	            values = Flux.Tracker.data(values)
	            probScore = Flux.Tracker.data(probScore)

	            # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
	            f2auc = pyauc(test[2] .- 1, probScore)
	            println("mem κ = $κ AUC f2: $f2auc")

				probScore = f3score(mem, model, test[1], κ)

	            # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
	            f3auc = pyauc(test[2] .- 1, probScore)
	            println("mem κ = $κ AUC f3: $f3auc")

	            push!(results, (ac, auc_pxv, f2auc, f3auc, values, probScore, ar, it, κ, "rnd"))
			end
        end
		exit()
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAESvaeProbMemLearning/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# datasets = ["breast-cancer-wisconsin", "sonar", "statlog-segment"]
# datasets = ["breast-cancer-wisconsin"]
# datasets = ["magic-telescope"]
datasets = ["pendigits"]
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

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), 1:10, batchSize, iterations, i)
	    # results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [128 512], [0], [1], [0.1 0.01 1 10 100], [0.1, 0.5, 0.9])
		results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [128], [0], [1], [0.2], [0.1])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
