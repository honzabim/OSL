include("bessel.jl")

import Flux
using NNlib
using Distributions
using SpecialFunctions
using Adapt
using Juno

Base.normalize(v) = v ./ (sqrt(sum(v .^ 2) + eps(Float32)))

struct SVAE
	q	# encoder (inference modul)
	g	# decoder (generator)
	zdim
	hue #Hyperspherical Uniform Entropy - constant with dimensionality - zdim
	μzfromhidden # function to compute μz from the hidden layer
	κzfromhidden # function to compute κz from the hidden layer

	"""
	SVAE(q, g, hdim, zdim) Constructor of the S-VAE with hidden dim `hdim` and latent dim = `zdim`. `zdim > 3`
	"""
	SVAE(q, g, hdim::Integer, zdim::Integer, T) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Dense(hdim, zdim, normalize)), Adapt.adapt(T, Dense(hdim, 1, softplus)))
end

Flux.treelike(SVAE)

vmfentropy(m, κ) = .-κ .* besselix(m / 2, κ) ./ besselix(m / 2 - 1, κ) .- ((m ./ 2 .- 1) .* log.(κ) .- (m ./ 2) .* log(2π) .- (κ .+ log.(besselix(m / 2 - 1, κ))))
huentropy(m) = m / 2 * log(π) + log(2) - lgamma(m / 2)

function loss(m::SVAE, x)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	z = μz
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + mean(kldiv(m, κz))
end

function zparams(model::SVAE, x)
	hidden = model.q(x)
	return model.μzfromhidden(hidden), model.κzfromhidden(hidden)
end

kldiv(model::SVAE, κ) = .- vmfentropy(model.zdim, κ) .+ model.hue
# kldiv(model::SVAE, κ::TrackedArray) = Tracker.track(kldiv, model, κ)
# kldiv(model::SVAE, κ::Flux.Tracker.TrackedReal) = Tracker.track(kldiv, model, κ)
#
# function ∇kldiv(model::SVAE, κ)
# 	m = model.zdim
# 	k = Flux.Tracker.data(κ)
#
# 	a = @. besselix(m / 2 + 1, k) / besselix(m / 2 - 1, k)
# 	b = @. besselix(m / 2, k) * (besselix(m / 2 - 2, k) + besselix(m / 2, k)) / besselix(m / 2 - 1, k) ^ 2
#
# 	return @. k / 2 * (a - b + 1)
# end
#
# Tracker.back(::typeof(kldiv), Δ, model::SVAE, κ) = Tracker.@back(κ, ∇kldiv(model, κ) .* Δ)


function sampleω(model::SVAE, κ)
	m = model.zdim
	c = @. sqrt(4κ ^ 2 + (m - 1) ^ 2)
	b = @. (-2κ + c) / (m - 1)
	a = @. (m - 1 + 2κ + c) / 4
	d = @. (4 * a * b) / (1 + b) - (m - 1) * log(m - 1)
	ω = rejectionsampling(m, a, b, d)
	return ω
end

function rejectionsampling(m, a, b, d)
	beta = Beta((m - 1) / 2, (m - 1) / 2)
	T = eltype(Flux.Tracker.data(a))
	ϵ, u = Adapt.adapt(T, rand(beta, size(Flux.Tracker.data(a))...)), Adapt.adapt(T, rand(size(Flux.Tracker.data(a))))

	accepted = isaccepted(ϵ, u, m, Flux.data(a), Flux.Tracker.data(b), Flux.data(d))
	while !all(accepted)
		mask = .! accepted
		ϵ[mask] = Adapt.adapt(T, rand(beta, sum(mask)))
		u[mask] = Adapt.adapt(T, rand(sum(mask)))
		accepted[mask] .= isaccepted(mask, ϵ, u, m, Flux.data(a), Flux.data(b), Flux.data(d))
	end
	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
end

isaccepted(mask, ϵ, u, m:: Int, a, b, d) = isaccepted(ϵ[mask], u[mask], m, a[mask], b[mask], d[mask])
function isaccepted(ϵ, u, m:: Int, a, b, d)
	ω = @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
	t = @. 2 * a * b / (1 - (1 - b) * ϵ)
	@. (m - 1) * log(t) - t + d >= log(u)
end

function householderrotation(zprime, μ)
	e1 = similar(μ) .= 0
	e1[1, :] .= 1
	u = e1 .- μ
	normalizedu = u ./ sqrt.(sum(u.^2, 1) + eps(Float32))
	return zprime .- 2 .* sum(zprime .* normalizedu, 1) .* normalizedu
end

function samplez(m::SVAE, μz, κz)
	ω = sampleω(m, κz)
	normal = Normal()
	v = Adapt.adapt(eltype(Flux.Tracker.data(κz)), rand(normal, size(μz, 1) - 1, size(μz, 2)))
	v = v ./ sqrt.(sum(v .^ 2, 1)) + eps(Float32)
	z = householderrotation(vcat(ω, sqrt.(1 .- ω .^ 2) .* v), μz)
	return z
end

zfromx(m::SVAE, x) = samplez(m, zparams(m, x)...)

gradtest(f, xs::AbstractArray...) = Flux.Tracker.gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(dims)...)
