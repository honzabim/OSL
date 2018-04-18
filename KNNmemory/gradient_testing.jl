using Flux, Flux.Tracker

mutable struct Lparam
    p::Float64
end

lossParam = Lparam(2.)


origW = param(rand(2, 5))
origb = param(rand(2))

W = origW
b = origb

predict(x) = W * x .+ b
loss(x, y) = sum((predict(x) .- y).^lossParam.p)

x, y = rand(5), rand(2) # Dummy data

l = loss(x, y)
back!(l)
g1 = W.grad

l = loss(x, y)
back!(l)
g2 = W.grad

W = origW
b = origb

l = loss(x, y)
back!(l)
g3 = W.grad

lossParam.p = 3.

l = loss(x, y)
back!(l)
g4 = W.grad

W = origW
b = origb
