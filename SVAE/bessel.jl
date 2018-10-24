using DiffRules
using SpecialFunctions
using PyCall
using Flux


# Julia seems to have all the Bessel functions biult-in. It just needed the Diff rule.

# function ∇besselix(v, x)
#     @pyimport sc.special as sc
#     if v == 0
#         return sc.i0e(x)
#     elseif v == 1
#         return sc.i1e(x)
#     else
#         return besselix(v - 1, x) - besselix(v, x) * (v + x) / x
#     end
# end

# DiffRules.@define_diffrule SpecialFunctions.besselix(ν, x) = :NaN, :(besselix($ν - 1, $x) - besselix($ν, $x) * ($ν + $x) / $x)
# DiffRules.@define_diffrule SpecialFunctions.besselix(ν, x) = :NaN, :(∇besselix($v, $x))

# @pyimport scipy.special as sc
#
# function mybesselix(ν, x)
#     results = similar(ν) .= 0
#     results[ν .== 0] = sc.i0e.(x[ν .== 0])
#     results[ν .== 1] = sc.i1e.(x[ν .== 1])
#     νother = @. ν != 0 & ν != 1
#     results[νother] = sc.ive.(ν[νother], x[νother])
#     return results
# end

mybesselix(ν, x) = besselix.(ν, x)

∇mybesselix(ν, x::Flux.Tracker.TrackedMatrix) = ∇mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesselix(ν, x::Flux.Tracker.TrackedReal) = ∇mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesselix(ν, x::Flux.Tracker.TrackedArray) = ∇mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesselix(ν, x) = @. mybesselix(ν - 1, x) - mybesselix(ν, x) * (ν + x) / x

SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(mybesselix, ν, x)
SpecialFunctions.besselix(ν::Real, x::Flux.Tracker.TrackedReal) = Flux.Tracker.track(mybesselix, ν, x)
SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedArray) = Flux.Tracker.track(mybesselix, ν, x)
SpecialFunctions.besselix(ν, x::AbstractArray) = besselix.(ν, x)

Flux.Tracker.@grad function mybesselix(ν, x)
    return mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x)), Δ -> (nothing, ∇mybesselix(ν, x) .* Δ)
end

mybesseli(ν, x) = besseli.(ν, x)

∇mybesseli(ν, x::Flux.Tracker.TrackedMatrix) = ∇mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesseli(ν, x::Flux.Tracker.TrackedReal) = ∇mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesseli(ν, x::Flux.Tracker.TrackedArray) = ∇mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesseli(ν, x) = @. (besseli(ν - 1, x) + besseli(ν + 1, x)) / 2

SpecialFunctions.besseli(ν, x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(mybesseli, ν, x)
SpecialFunctions.besseli(ν::Real, x::Flux.Tracker.TrackedReal) = Flux.Tracker.track(mybesseli, ν, x)
SpecialFunctions.besseli(ν, x::Flux.Tracker.TrackedArray) = Flux.Tracker.track(mybesseli, ν, x)
SpecialFunctions.besseli(ν, x::AbstractArray) = besseli.(ν, x)

Flux.Tracker.@grad function mybesseli(ν, x)
    return mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x)), Δ -> (nothing, ∇mybesseli(ν, x) .* Δ)
end
