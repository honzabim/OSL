using DiffRules
using SpecialFunctions

DiffRules.@define_diffrule SpecialFunctions.besselix(ν, x)   = :NaN, :(besselix($ν - 1, $x) - besselix($ν, $x) * (v + z) / z)
