using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using FluxExtensions

push!(LOAD_PATH, pwd())
using KNNmem

imgs = MNIST.images()
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()
Y = labels
oneHotY = Flux.onehot(Y, 0:9) # for softmax

tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = MNIST.labels(:test)
oneHotTY = Flux.onehot(tY, 0:9);

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()
# One-hot-encode the labels
#Y = onehotbatch(labels, 0:9)
Y = labels

m = Chain(
  FluxExtensions.ResDense(28^2, 32, relu),
  FluxExtensions.ResDense(32, 10, relu))

memory = KNNmemory(1000, 10, 256, 10)

loss(x, y) = trainQuery!(memory, m(x), y)
accuracy(x, y) = mean(query(memory, m(x)) .== y)

opt = ADAM(params(m))

iterations = 1000
batchSize = 100

accuracy(X, Y)

for i in 1:iterations
    # sample from data
    batchIndeces = rand(1:size(X, 2),batchSize)
    x = X[:, batchIndeces];
    y = Y[batchIndeces];

    # gradient computation and update
    l = loss(x, y)
    if i % 100 == 0
        println(l)
    end
    Flux.Tracker.back!(l)
    opt()
end

# Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

accuracy(X, Y)

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = MNIST.labels(:test)

accuracy(tX, tY)
