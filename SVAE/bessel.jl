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
# function mybessel(ν, x)
#     results = similar(ν) .= 0
#     results[ν .== 0] = sc.i0e.(x[ν .== 0])
#     results[ν .== 1] = sc.i1e.(x[ν .== 1])
#     νother = @. ν != 0 & ν != 1
#     results[νother] = sc.ive.(ν[νother], x[νother])
#     return results
# end

mybessel(ν, x) = besselix.(ν, x)

∇mybessel(ν, x::Flux.Tracker.TrackedMatrix) = ∇mybessel(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybessel(ν, x::Flux.Tracker.TrackedReal) = ∇mybessel(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybessel(ν, x::Flux.Tracker.TrackedArray) = ∇mybessel(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybessel(ν, x) = @. mybessel(ν - 1, x) - mybessel(ν, x) * (ν + x) / x

SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(mybessel, ν, x)
SpecialFunctions.besselix(ν::Real, x::Flux.Tracker.TrackedReal) = Flux.Tracker.track(mybessel, ν, x)
SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedArray) = Flux.Tracker.track(mybessel, ν, x)
SpecialFunctions.besselix(ν, x::AbstractArray) = besselix.(ν, x)
Flux.Tracker.back(::typeof(mybessel), Δ, ν, x) = Flux.Tracker.back(x, ∇mybessel(ν, Flux.data(x)) .* Δ)
