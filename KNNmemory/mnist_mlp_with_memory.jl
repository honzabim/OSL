using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using FluxExtensions

push!(LOAD_PATH, "/home/jan/dev/OSL/KNNmemory")
using KNNmem

imgs = MNIST.images()
X = convert(Array{Float32}, hcat((reshape.(imgs, :))...))



labels = MNIST.labels()
Y = labels

tX = convert(Array{Float32}, hcat(float.(reshape.(MNIST.images(:test), :))...))
tY = MNIST.labels(:test)

m = Flux.adapt(Float32, Chain(
  FluxExtensions.ResDense(28^2, 32, relu),
  FluxExtensions.ResDense(32, 10, relu)))

memory = KNNmemory{Float32}(1000, 10, 256, 10)

loss(x, y) = trainQuery!(memory, m(x), y)
function accuracy(x, y)
    (vals, _) = query(memory, m(x))
    mean(vals .== y)
end

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

accuracy(tX, tY)
