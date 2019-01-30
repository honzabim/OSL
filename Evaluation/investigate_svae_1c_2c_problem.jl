using Adapt
using Flux
using MLBase: roc, f1score, precision, recall
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using UCI

folderpath = "D:/dev/julia/"
push!(LOAD_PATH, folderpath)
using NearestNeighbors
using StatsBase
using InformationMeasures
using Random

using Plots
plotly()

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_anom.jl")

function createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_anom(encoder, decoder, hiddenDim, latentDim, T)

    learnRepresentation!(data, foo) = wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
	learnPrintingRepresentation!(data, foo) = printingwloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    learnAnomaly!(μ) = set_anomalous_μ(svae, μ)
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
data = normalizecolumns(randn(2, 300)) .* 2 .+ randn(2, 300) .* 0.3
labels = (([norm(data[:, i]) for i in 1:size(data, 2)] .> 2.5) .| ([norm(data[:, i]) for i in 1:size(data, 2)] .< 1.5)) .* 1
newclass = randn(2, 30) .* 0.2

nrm = labels .== 0
anm = labels .== 1

scatter(data[1, nrm], data[2, nrm], label = "Normal")
scatter!(data[1, anm], data[2, anm], label = "Anomalous")
scatter!(newclass[1, :], newclass[2, :], label = "New Class")
display(plot!(size = (950, 900), title = "Dataset"))

inputDim = 2
hiddenDim = 32
numLayers = 3
latentDim = 3
nonlinearity = "relu"
layerType = "Dense"
batchSize = 100
numBatches = 10000
βsvae = 0.5

(svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesWass!, learnWithAnomaliesPrintingWass!) = createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, βsvae)
learnAnomaly!([-1., 0, 0])

# opt = Flux.Optimise.ADAM(Flux.params(svae), 1e-4)
# cb = Flux.throttle(() -> println("SVAE : $(learnPrintingRepresentation!(data, []))"), 5)
# Flux.train!(learnRepresentation!, RandomBatches((data, zeros(labels)), batchSize, numBatches), opt, cb = cb)

opt = Flux.Optimise.ADAM(Flux.params(svae), 3e-5)
cb = Flux.throttle(() -> println("SVAE : $(learnWithAnomaliesWass!(data, zero(labels)))"), 5)
Flux.train!(learnWithAnomaliesWass!, RandomBatches((data, zero(labels)), batchSize, numBatches), opt, cb = cb)

fillc = true
nlevels = 20

x = y = -3:0.2:3
svaescore = (x, y) -> -pxvita(svae, [x, y])[1]
csvae = contour(x, y, svaescore, fill = fillc, levels = nlevels)
display(plot(csvae, title = "P(X) Vita", size = (600, 550)))

z = Flux.Tracker.data(zparams(svae, data)[1])
nz = Flux.Tracker.data(zparams(svae, newclass)[1])
scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous")
plot!(scatter3d!(nz[1, :], nz[2, :], nz[3, :], label = "New Class"), size = (600, 550))
μnormal = normalize(vec(mean(z, dims = 2)))
μplot = μnormal .* 1.3
display(Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], color = "red", linewidth = "5"))
μanom = Flux.Tracker.data(svae.anom_priorμ)
display(Plots.plot!([0, μanom[1]], [0, μanom[2]], [0, μanom[3]], color = "green", linewidth = "5"))

println("κ = svae.anom_priorκ")

z = Flux.Tracker.data(samplez(svae, zparams(svae, data)...))
nz = Flux.Tracker.data(samplez(svae, zparams(svae, newclass)...))
scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous")
plot!(scatter3d!(nz[1, :], nz[2, :], nz[3, :], label = "New Class"), size = (600, 550))
μnormal = normalize(vec(mean(z, dims = 2)))
μplot = μnormal .* 1.3
display(Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], title = "sampled", color = "red", linewidth = "5"))


numBatches = 1000
learnAnomaly!(.-μnormal)
opt = Flux.Optimise.ADAM(Flux.params(svae), 1e-4)
cb = Flux.throttle(() -> println("SVAE : $(learnWithAnomaliesPrintingWass!(hcat(data, newclass), vcat(zero(labels), ones(size(newclass, 2)))))"), 5)

for i in 1:5
	Flux.train!(learnWithAnomaliesWass!, RandomBatches((hcat(data, newclass), vcat(zero(labels), ones(size(newclass, 2)))), batchSize, numBatches), opt, cb = cb)

	z = Flux.Tracker.data(zparams(svae, data)[1])
	nz = Flux.Tracker.data(zparams(svae, newclass)[1])
	scatter3d(z[1, nrm], z[2, nrm], z[3, nrm], label = "Normal")
	scatter3d!(z[1, anm], z[2, anm], z[3, anm], label = "Anomalous")
	plot!(scatter3d!(nz[1, :], nz[2, :], nz[3, :], label = "New Class"), size = (600, 550))
	μnormal = normalize(vec(mean(z, dims = 2)))
	μplot = μnormal .* 1.3
	Plots.plot!([0, μplot[1]], [0, μplot[2]], [0, μplot[3]], color = "red", linewidth = "5")
	μanom = Flux.Tracker.data(svae.anom_priorμ)
	display(Plots.plot!([0, μanom[1]], [0, μanom[2]], [0, μanom[3]], color = "green", linewidth = "5"))
end
