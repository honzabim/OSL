using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO

# folderpath = "D:/dev/julia/"
folderpath = "/home/bimjan/dev/julia/"
# folderpath = "D:/dev/"

push!(LOAD_PATH, folderpath, folderpath * "OSL/KNNmemory/")
# push!(LOAD_PATH, folderpath)
using KNNmem
using FluxExtensions
using ADatasets
using NearestNeighbors
using StatsBase
using InformationMeasures
using Random
using LinearAlgebra
using CSV
using DataFrames

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

# using Plots
# plotly()

include(folderpath * "OSL/SVAE/svae.jl")

csvOutputFolder = folderpath * "data/CSV/"
mkpath(csvOutputFolder)


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
    mem, train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, memorySize, labelCount, α, T) # TODO the second memsize should be k !!!

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

    return mem, svae, learnRepresentation!, learnAnomaly!, classify, justTrain!
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
	prob = exp.(log_normal(Flux.Tracker.data(model.g(memory.M')), repeat(x, 1, size(memory.M, 1))) .+ log_vmf_c(memory.M', z, κ))
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

function f3normalscore(memory::KNNmemory, model::SVAE, z::Vector, x::Vector, κ)
	sum(exp.(log_normal(Flux.Tracker.data(model.g(memory.M')), repeat(x, 1, size(memory.M, 1))) .+ log_vmf_c(memory.M', z, κ)))
end

function f3normalscore(memory::KNNmemory, model::SVAE, x, κ)
	(μz, _) = zparams(model, x)
	μz = Flux.Tracker.data(μz)
	score = []
	for i in 1:size(x, 2)
		push!(score, f3normalscore(memory, model, μz[:, i], Flux.Tracker.data(x[:, i]), κ))
	end
	return score
end

function savetodf(ds, filename)
    header = vcat("label", (repeat(["X"], size(ds[1], 1)) .* string.(collect(1:size(ds[1], 1)))...))
    labels = (x -> x == 1 ? "nominal" : "anomaly").(ds[2])
    data = hcat(labels, collect(ds[1]'))
    df = DataFrame(data)
    rename!(df, :x1 => :label)
    CSV.write(filename, df)
end


function runExperiment(datasetName, train, test, createModel, feedbackCounts, batchSize, numBatches, it)
    anomalyRatios = [0.05]
    results = []
    for ar in anomalyRatios
        println("Running $datasetName with ar: $ar iteration: $it")
        (mem, model, learnRepresentation!, learnAnomaly!, classify, justTrain!) = createModel()
        opt = Flux.Optimise.ADAM(Flux.params(model), 4e-5)
        cb = Flux.throttle(() -> println("$datasetName AR=$ar : $(justTrain!(train[1], []))"), 5)
        Flux.train!(justTrain!, RandomBatches((train[1], zero(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)
        # FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2] .- 1), batchSize, numBatches), ()->(), 100)
		println("Finished learning $datasetName with ar: $ar iteration: $it")

        learnRepresentation!(train[1], zero(train[2]))

		pxv = collect(.-pxvita(model, test[1])')
		auc_pxv = pyauc(test[2] .- 1, pxv)
		println("P(X) Vita AUC = $auc_pxv on $datasetName with ar: $ar iteration: $it")

		for κ in [0.5, 1, 3, 5, 10, 50]
			# reset the memory
			mem.M = collect(normalizecolumns(randn(size(mem.M')))')
			mem.V = zeros(Int, size(mem.V))
			learnRepresentation!(train[1], zero(train[2]))
			anomalies_discovered = 0

			t = copy(train[1])
			l = copy(train[2])
			for fc in feedbackCounts

				mostAnomalousId = -1
				avgNormal = collect(normalize(vec(mean(zparams(model, t)[1], dims = 2)))')

				if ! any(mem.V .== 1)
					mostAnomalousId = argmax(vec((.-(avgNormal) * zparams(model, t)[1])))
					# mostAnomalousId = argmax(.-vec(f3normalscore(mem, model, t, κ)))
					# println("Avg chose most anomalous with label $(l[mostAnomalousId] - 1)")
					# mostAnomalousId = argmax(vec(.-pxvita(model, test[1])))
					# println("Pxvita chose most anomalous with label $(train[2][mostAnomalousId] - 1)")
				else
					# mostAnomalousId = argmax(vec(classify(t, κ)[2]))
					values, probScore = classify(t, κ)
					mostAnomalousId = argmax(vec(f3score(mem, model, t[:, values .== 0], κ)))
				end

				data = Flux.Tracker.data(zparams(model, train[1])[1])
				nrm = train[2] .== 1
				anm = train[2] .== 2

				# scatter3d(data[1, nrm], data[2, nrm], data[3, nrm], label = "Normal")
				# scatter3d!(data[1, anm], data[2, anm], data[3, anm], label = "Anomalous")
				# scatter3d!(mem.M[mem.V .== 0, 1], mem.M[mem.V .== 0, 2], mem.M[mem.V .== 0, 3], label = "Memory normal")
				# scatter3d!(mem.M[mem.V .== 1, 1], mem.M[mem.V .== 1, 2], mem.M[mem.V .== 1, 3], label = "Memory anomalous")
				# plot!(size = (950, 900), title = "Dataset")
				# display(Plots.plot!([0, -avgNormal[1]], [0, -avgNormal[2]], [0, -avgNormal[3]], color = "red", linewidth = "5"))

				# println("We chose most anomalous with label $(l[mostAnomalousId] - 1)")
				learnAnomaly!(t[:, mostAnomalousId], l[mostAnomalousId] - 1)
				if l[mostAnomalousId] - 1 == 1
					anomalies_discovered += 1
				end
				println("Feedback count $fc κ $κ anoms_disc: $anomalies_discovered")
				# println("So far we hit $anomalies_discovered anomalies out of $fc trials")

				t = hcat(t[:, 1:(mostAnomalousId - 1)], t[:, (mostAnomalousId + 1):end])
				deleteat!(l, mostAnomalousId)

	            values, probScore = classify(test[1], κ)
	            values = Flux.Tracker.data(values)
	            probScore = Flux.Tracker.data(probScore)

	            # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
	            f2auc = pyauc(test[2] .- 1, probScore)
	            # println("mem κ = $κ AUC f2: $f2auc")

				probScore = f3score(mem, model, test[1], κ)

	            # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
	            f3auc = pyauc(test[2] .- 1, probScore)
	            # println("mem κ = $κ AUC f3: $f3auc")

	            push!(results, (fc, auc_pxv, f2auc, f3auc, ar, it, κ, anomalies_discovered))
			end
		end
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/WSVAEvsTDnewacqAUC/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
# datasets = ["breast-cancer-wisconsin", "sonar", "statlog-segment"]
datasets = ["abalone"]
# datasets = ["pendigits"]
difficulties = ["easy"]
const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 13000

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for i in 1:5
	for (dn, df) in zip(datasets, difficulties)

	    trainall, test, clusterdness = loadData(dn, df)

		ar = 0.05
		train = ADatasets.subsampleanomalous(trainall, ar)
		savetodf(train, csvOutputFolder * dn * "-$ar-$i.csv")

	    println("$dn")
	    println("$(size(train[2]))")
	    println("$(counts(train[2]))")
	    println("Running svae...")

	    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), 1:50, batchSize, iterations, i)
	    results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [128 256], [0], [1], [.1 1 5])
	    results = reshape(results, length(results), 1)
	    save(outputFolder * dn *  "-$i-svae.jld2", "results", results)
	end
end
