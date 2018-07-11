using DiffRules
using SpecialFunctions
using PyCall


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

mybessel(ν, x) = SpecialFunctions.besselix.(ν, x)
∇mybessel(ν,x) = @. SpecialFunctions.besselix(ν - 1, x) - SpecialFunctions.besselix(ν, x) * (ν + x) / x

SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(mybessel, ν, x)
Flux.Tracker.back(::typeof(mybessel), Δ, ν, x) = Flux.Tracker.@back(x, ∇mybessel(ν,Flux.data(x)) .* Δ)
