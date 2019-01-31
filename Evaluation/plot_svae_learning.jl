using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using ADatasets
using UCI

folderpath = "D:/dev/julia/"
push!(LOAD_PATH, folderpath)
using NearestNeighbors
using StatsBase
using InformationMeasures
using Random

using Plots
plotlyjs()

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_anom.jl")

const dataPath = folderpath * "data/loda/public/datasets/numerical"
loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

function createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_anom(encoder, decoder, hiddenDim, latentDim, T)

    learnRepresentation!(data, foo) = wlossprior(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
	learnPrintingRepresentation!(data, foo) = printingwloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    learnAnomaly!(μ) = set_anomalous_μ_nonparam(svae, μ)
	learnWithAnomaliesWass!(data, labels) = wloss_anom_vasek(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1), 0.5)
	learnWithAnomaliesPrintingWass!(data, labels) = printing_wloss_anom_vasek(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))

    return svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesWass!, learnWithAnomaliesPrintingWass!
end

# Create data
# Gauss
# data = randn(2, 300)
# labels = ([norm(data[:, i]) for i in 1:size(data, 2)] .> 2.7) .* 1
# newclass = randn(2, 50) ./ 8 .+ 0.75

# Donut
# data = normalizecolumns(randn(2, 300)) .* 2 .+ randn(2, 300) .* 0.3
# labels = (([norm(data[:, i]) for i in 1:size(data, 2)] .> 2.5) .| ([norm(data[:, i]) for i in 1:size(data, 2)] .< 1.5)) .* 1
# newclass = randn(2, 30) .* 0.2

# nrm = labels .== 0
# anm = labels .== 1

# scatter(data[1, nrm], data[2, nrm], label = "Normal")
# scatter!(data[1, anm], data[2, anm], label = "Anomalous")
# scatter!(newclass[1, :], newclass[2, :], label = "New Class")
# display(plot!(size = (950, 900), title = "Dataset"))

dataset = "breast-cancer-wisconsin"
difficulty = "easy"

train, test, clusterdness = loadData(dataset, difficulty)

inputDim = size(train[1], 1)
hiddenDim = 32
numLayers = 3
latentDim = 3
nonlinearity = "relu"
layerType = "Dense"
batchSize = 100
numBatches = 10000
βsvae = 0.5

(svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesWass!, learnWithAnomaliesPrintingWass!) = createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, βsvae)
learnAnomaly!(normalize([-1., -1, 0]))

# z = Flux.Tracker.data(zparams(svae, train[1])[1])
z = Flux.Tracker.data(zfromx(svae, train[1]))
nrm = train[2] .== 1
anm = train[2] .== 2
bp = scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], alpha = 0.8, label = "Normal")
bp = plot!(scatter3d!(z[1, anm], z[2, anm], z[3, anm], alpha = 0.8,  label = "Anomalous"), size = (600, 550), title = "Train - before")

opt = Flux.Optimise.ADAM(Flux.params(svae), 3e-5)
cb = Flux.throttle(() -> println("SVAE : $(learnRepresentation!(train[1], zero(train[2])))"), 5)
Flux.train!(learnRepresentation!, RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

# z = Flux.Tracker.data(zparams(svae, train[1])[1])
z = Flux.Tracker.data(zfromx(svae, train[1]))
nrm = train[2] .== 1
anm = train[2] .== 2
ap = scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
ap = plot!(scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous"), size = (600, 550))
μnormal = normalize(vec(mean(z, dims = 2)))
μplot = μnormal .* 1.3
ap = Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], color = "red", linewidth = "5", label = "learned mean")
μanom = Flux.Tracker.data(svae.anom_priorμ)
ap = Plots.plot!([0, μanom[1]], [0, μanom[2]], [0, μanom[3]], color = "green", linewidth = "5", label = "prior mean", title = "Train - after")

z = Flux.Tracker.data(zfromx(svae, test[1]))
nrm = test[2] .== 1
anm = test[2] .== 2
tp = scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
tp = plot!(scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous"), size = (600, 550))
μnormal = normalize(vec(mean(z, dims = 2)))
μplot = μnormal .* 1.3
tp = Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], color = "red", linewidth = "5", label = "learned mean")
μanom = Flux.Tracker.data(svae.anom_priorμ)
tp = Plots.plot!([0, μanom[1]], [0, μanom[2]], [0, μanom[3]], color = "green", linewidth = "5", label = "prior mean", title = "Test - after")
display(plot(bp))
display(plot(ap))
display(plot(tp))
ext = "svg"
savefig(bp, "figures/before.$ext")
savefig(ap, "figures/after.$ext")
savefig(tp, "figures/test.$ext")

# println("κ = svae.anom_priorκ")
#
# z = Flux.Tracker.data(samplez(svae, zparams(svae, data)...))
# nz = Flux.Tracker.data(samplez(svae, zparams(svae, newclass)...))
# scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
# scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous")
# plot!(scatter3d!(nz[1, :], nz[2, :], nz[3, :], label = "New Class"), size = (600, 550))
# μnormal = normalize(vec(mean(z, dims = 2)))
# μplot = μnormal .* 1.3
# display(Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], title = "sampled", color = "red", linewidth = "5"))
#
#
# numBatches = 1000
# learnAnomaly!(.-μnormal)
# opt = Flux.Optimise.ADAM(Flux.params(svae), 1e-4)
# cb = Flux.throttle(() -> println("SVAE : $(learnWithAnomaliesPrintingWass!(hcat(data, newclass), vcat(zero(labels), ones(size(newclass, 2)))))"), 5)
#
# for i in 1:5
# 	Flux.train!(learnWithAnomaliesWass!, RandomBatches((hcat(data, newclass), vcat(zero(labels), ones(size(newclass, 2)))), batchSize, numBatches), opt, cb = cb)
#
# 	z = Flux.Tracker.data(zparams(svae, data)[1])
# 	nz = Flux.Tracker.data(zparams(svae, newclass)[1])
# 	scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
# 	scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous")
# 	plot!(scatter3d!(nz[1, :], nz[2, :], nz[3, :], label = "New Class"), size = (600, 550))
# 	μnormal = normalize(vec(mean(z, dims = 2)))
# 	μplot = μnormal .* 1.3
# 	Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], color = "red", linewidth = "5")
# 	μanom = Flux.Tracker.data(svae.anom_priorμ)
# 	display(Plots.plot!([0, μanom[1]], [0, μanom[2]], [0, μanom[3]], color = "green", linewidth = "5"))
# end
