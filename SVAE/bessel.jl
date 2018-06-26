using DiffRules
using SpecialFunctions

'''
Julia seems to have all the Bessel functions biult-in. It just needed the Diff rule.
'''
DiffRules.@define_diffrule SpecialFunctions.besselix(ν, x)   = :NaN, :(besselix($ν - 1, $x) - besselix($ν, $x) * ($ν + $x) / $x)
