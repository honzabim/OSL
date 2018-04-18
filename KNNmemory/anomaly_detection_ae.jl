using Flux
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, false_negative_rate

push!(LOAD_PATH, pwd(), "/home/jan/dev/anomaly detection/anomaly_detection/src", "/home/jan/dev/FluxExtensions.jl/src")
using KNNmem
using AnomalyDetection

dataPath = "/home/jan/dev/data/loda/public/datasets/numerical"
allData = AnomalyDetection.loaddata(dataPath)

dataset = allData["pendigits"]
train, test, clusterdness = AnomalyDetection.makeset(dataset, 0.9, "normal", 0.05, "high")

inputSize = size(train.data, 1)
zSize = 10 # size of the encoding vector
memorySize = 100 # size of the kNN memory
k = 16 # number of the nearest neighbors in the kNN
β = 0.1 # parameter setting the weight of the memory loss

# RELU could easily create a zero vector which creates problems in the memory
# as the memory demands vectors with a norm = 1 and normalizes the others
encoder = Dense(inputSize, zSize, tanh)
decoder = Dense(zSize, inputSize, tanh)
model = Chain(encoder, decoder)

memory = KNNmemory(memorySize, zSize, k, 2)

reconstructionError(x) = Flux.mse(model(x), x)
loss(x, y) = reconstructionError(x) + β * trainQuery!(memory, encoder(x), y)
opt = ADAM(params(model))

iterations = 1000
batchSize = 1000

for i in 1:iterations
    # sample from data
    batchIndeces = rand(1:size(train.data, 2),batchSize)
    x = train.data[:, batchIndeces]
    y = train.labels[batchIndeces]

    # gradient computation and update
    l = loss(x, y)
    if i % 100 == 0
        println(l)
    end
    Flux.Tracker.back!(l)
    opt()
end

println("Reconstruction err: $(reconstructionError(test.data))")

rocData = roc(test.labels, query(memory, encoder(test.data)))
print(rocData)
print("precision: $(precision(rocData))\n")
print("f1score: $(f1score(rocData))\n")
print("recall: $(recall(rocData))\n")
print("false positive rate: $(false_positive_rate(rocData))\n")
print("equal error rate: $((false_positive_rate(rocData) + false_negative_rate(rocData))/2)\n")
