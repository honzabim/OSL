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

using PyCall
@pyimport sklearn.metrics as sm

function pyauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = true)
	pyauc = sm.auc(pyfpr, pytpr)
end

include(folderpath * "OSL/SVAE/svae_anom.jl")
include(folderpath * "OSL/SVAE/svae_vamp.jl")

function createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_anom(encoder, decoder, hiddenDim, latentDim, T)

    learnRepresentation!(data, foo) = wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
	learnPrintingRepresentation!(data, foo) = printingwloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    learnAnomaly!(anomaly) = set_anomalous_hypersphere(svae, anomaly)
	learnWithAnomaliesLkh!(data, labels) = wloss_anom_lkh(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))
	learnWithAnomaliesWass!(data, labels) = wloss_anom_wass(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))

    return svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!
end

function createSVAE_vamp(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAE_vamp(encoder, decoder, hiddenDim, latentDim, 5, T)

    learnRepresentation!(data, foo) = wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
	learnPrintingRepresentation!(data, foo) = printingwloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
    learnAnomaly!(anomaly) = set_anomalous_hypersphere(svae, anomaly)
	learnWithAnomaliesLkh!(data, labels) = wloss_anom_lkh(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))
	learnWithAnomaliesWass!(data, labels) = wloss_anom_wass(svae, data, labels, β, (x, y) -> mmd_imq(x, y, 1))

    return svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!
end


# dataset = "breast-cancer-wisconsin"
# class_label = ""
# dataset = "libras"
# class_label = "1-6"
# dataset = "letter-recognition"
# class_label = "U-R"
dataset = "glass"
class_label = "2-6"

data, normal_labels, anomaly_labels = UCI.get_umap_data(dataset)
subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
dataid = find(x -> x[2] == class_label, subdatasets)
subdataset = subdatasets[dataid]
_X_tr, _y_tr, _X_tst, _y_tst = UCI.split_data(subdataset[1][1], 0.8)
_X_tr .-= minimum(_X_tr)
_X_tr ./= maximum(_X_tr)
_X_tst .-= minimum(_X_tst)
_X_tst ./= maximum(_X_tst)
train = (_X_tr, _y_tr)
test = (_X_tst, _y_tst)

inputDim = size(train[1], 1)
hiddenDim = 32
numLayers = 3
latentDim = 3
nonlinearity = "relu"
layerType = "Dense"
batchSize = 100
numBatches = 10000
βsvae = 0.5
βvamp = 0.5

(svae, learnRepresentation!, learnPrintingRepresentation!, learnAnomaly!, learnWithAnomaliesLkh!, learnWithAnomaliesWass!) = createSVAE_anom(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, βsvae)
(vamp, vlearnRepresentation!, vlearnPrintingRepresentation!, vlearnAnomaly!, vlearnWithAnomaliesLkh!, vlearnWithAnomaliesWass!) = createSVAE_vamp(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, βvamp)

opt = Flux.Optimise.ADAM(Flux.params(svae), 3e-5)
cb = Flux.throttle(() -> println("SVAE : $(learnPrintingRepresentation!(train[1], []))"), 5)
Flux.train!(learnRepresentation!, RandomBatches((train[1], zeros(train[2])), batchSize, numBatches), opt, cb = cb)

ascore = Flux.Tracker.data(.-pxvita(svae, test[1]))
auc = pyauc(test[2], ascore')
println(size(ascore))
println(size(test[2]))
println("AUC svae: $auc")

opt = Flux.Optimise.ADAM(Flux.params(vamp), 3e-5)
cb = Flux.throttle(() -> println("VAMP : $(vlearnPrintingRepresentation!(train[1], []))"), 5)
Flux.train!(vlearnRepresentation!, RandomBatches((train[1], zeros(train[2])), batchSize, numBatches), opt, cb = cb)

ascore = Flux.Tracker.data(.-pxvita(vamp, test[1]))
auc = pyauc(test[2], ascore')
println(size(ascore))
println(size(test[2]))
println("AUC vamp: $auc")

using Plots
plotly()

fillc = true
nlevels = 20

x = y = -4:0.2:4
svaescore = (x, y) -> -pxvita(svae, [x, y])[1]
vampscore = (x, y) -> -pxvita(vamp, [x, y])[1]
csvae = contour(x, y, svaescore, fill = fillc, levels = nlevels)
cvamp = contour(x, y, vampscore, fill = fillc, levels = nlevels)
display(plot(csvae, cvamp, layout = 2, title = ["SVAE" "VAMP"]))
