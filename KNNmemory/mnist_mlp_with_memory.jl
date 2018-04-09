using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated

include("KNNmemory.jl")

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()
# One-hot-encode the labels
#Y = onehotbatch(labels, 0:9)
Y = labels

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10))

memory = KNNmemory(1000, 10, 256, 10)

loss(x, y) = trainQuery!(memory, m(x), y)

accuracy(x, y) = mean(query(memory, m(x)) .== y)

dataset = repeated((X, Y), 100)
evalcb = () -> @show(loss(X, Y))
opt = ADAM(params(m))

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

accuracy(X, Y)

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

accuracy(tX, tY)
