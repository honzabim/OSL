using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using FluxExtensions
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, false_negative_rate

push!(LOAD_PATH, ".", "/home/jan/dev/anomaly detection/anomaly_detection/src")
using KNNmem
using AnomalyDetection

include("autoencoder_with_memory.jl");



# Prepare data

dataPath = "/home/jan/dev/data/loda/public/datasets/numerical"
allData = AnomalyDetection.loaddata(dataPath)

dataset = allData["abalone"]
train, test, clusterdness = AnomalyDetection.makeset(dataset, 0.9, "easy", 0.1, "high")
inputSize = size(train.data, 1)
zSize = 2

function sample(batchSize::Integer, data::AbstractArray{T, 2} where T, labels::AbstractArray{T, 1} where T)
    batchIndeces = rand(1:size(data, 2), batchSize)
    x = data[:, batchIndeces];
    y = labels[batchIndeces];
    return x, y
end

# Define the autoencoder

encoder = Chain(
  FluxExtensions.ResDense(inputSize, inputSize, relu),
  FluxExtensions.ResDense(inputSize, zSize, relu))

decoder = Chain(
  FluxExtensions.ResDense(zSize, inputSize, relu),
  FluxExtensions.ResDense(inputSize, inputSize, relu))

memorySize = 500
k = 32
labelCount = 2

ae = AutoencoderWithMemory(encoder, decoder, memorySize, zSize, k, labelCount)
optimizer = ADAM(params(Chain(encoder, decoder)))

anomaliesCount = sum(train.labels .== 1)
anomaliesIDs = (length(train.labels) - anomaliesCount + 1):length(train.labels)
regularDataIDs = 1:(length(train.labels) - anomaliesCount)

iterations = 1000
batchSize = 100

printInterationCount = 100
# Learn representations

for i in 1:iterations
    # sample from data
    (x, y) = sample(batchSize, train.data[:, regularDataIDs], train.labels[regularDataIDs])

    # gradient computation and update
    l = reconstructionError(ae, x)

    # teach the memory the regular data representation but don't use the loss
    memoryTrainQuery(ae, x, y)

    Flux.Tracker.back!(l)
    optimizer()

    if i % printInterationCount == 0
        println(Flux.Tracker.data(l))
    end
end

anomalyCounter = 1
for i in anomaliesIDs
    # show an anomaly to the memory
    l = learnAnomaly(ae, train.data[:, i], train.labels[i])
    Flux.Tracker.back!(l)
    optimizer()

    # test how well it got it
    print("\n\nAnomalies seen: $anomalyCounter \n")
    rocData = roc(test.labels, memoryClassify(ae, test.data))
    print(rocData)
    print("precision: $(precision(rocData))\n")
    print("f1score: $(f1score(rocData))\n")
    print("recall: $(recall(rocData))\n")
    print("false positive rate: $(false_positive_rate(rocData))\n")
    print("equal error rate: $((false_positive_rate(rocData) + false_negative_rate(rocData))/2)\n")
    anomalyCounter += 1
end
