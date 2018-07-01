include("bessel.jl")

import Flux
using NNlib
using Distributions
using SpecialFunctions

Base.normalize(v) = v ./ (sqrt(sum(v .^ 2) + eps(Float32)))

struct SVAE
	q	# encoder (inference modul)
	g	# decoder (generator)
	zdim
	μzfromhidden # function to compute μz from the hidden layer
	κzfromhidden # function to compute κz from the hidden layer

	"""
	SVAE(q, g, hdim, zdim) Constructor of the S-VAE with hidden dim `hdim` and latent dim = `zdim`. `zdim > 3`
	"""
	SVAE(q, g, hdim::Integer, zdim::Integer) = new(q, g, zdim, Dense(hdim, zdim, normalize), Dense(hdim, 1, softplus))
end

function loss(m::SVAE, x)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + mean(kldiv(m, κz))
end

function zparams(model::SVAE, x)
	_zparams = model.q(x)
	return model.μzfromhidden(_zparams), model.κzfromhidden(_zparams)
end

function kldiv(model::SVAE, κ)
	m = model.zdim
	return @. κ * besselix(m / 2, κ) / besselix(m / 2 - 1, κ) + (m / 2 - 1) * log(κ) - (m / 2) * log(2π) - (κ * log(besselix(m / 2 - 1, κ))) + m / 2 * log(π) + log(2) - lgamma(m / 2)
end

function sampleω(model::SVAE, κ)
	m = model.zdim
	c = @. √(4κ ^ 2 + (m - 1) ^ 2)
	b = @. (-2κ + c) / (m - 1)
	a = @. (m - 1 + 2κ + c) / 4
	d = @. (4 * a * b) / (1 + b) - (m - 1) * log(m - 1)
	ω = Flux.Tracker.collect(map((a, b, d) -> rejectionsampling(m, a, b, d), a, b, d))
	return ω
end

function rejectionsampling(m, a, b, d)
	uniform = Uniform()
	beta = Beta((m - 1) / 2., (m - 1) / 2.)
	ω = zero(a)
	while true
		ϵ = convert(eltype(Flux.Tracker.data(a)), rand(beta))
		ω = (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
		t = 2 * a * b / (1 - (1 - b) * ϵ)
		u = convert(eltype(Flux.Tracker.data(a)), rand(uniform))
		if Flux.Tracker.data(((m - 1) * log(t) - t + d)) >= log(u)
			break
		end
	end
	return ω
end

matrixtocolumns(x) = [x[:, i] for i in 1:size(x, 2)]

function householderrotation(zprime, μ)
	println("μ size $(size(μ))")
	e1 = zeros(eltype(Flux.Tracker.data(μ)), size(μ))
	e1[1, :] = 1
	# I would use mapslices but that does not work with Flux arrays - cannot create empty array
	u = e1 .- μ
	U = matrixtocolumns(u)
	Z = matrixtocolumns(zprime)
	NU = map(x -> normalize(x), U)
	z = Flux.Tracker.collect(hcat(map((u, z) -> z - 2 * (u' * z) * u, NU, Z)...))
	return z
end

function samplez(m::SVAE, μz, κz)
	ω = sampleω(m, κz)
	normal = Normal()
	v = Adapt.adapt(eltype(Flux.Tracker.data(κz)), rand(normal, size(μz, 1) - 1, size(μz, 2)))
	z = householderrotation(vcat(ω, sqrt.(1 .- ω .^ 2) .* v), μz)
	return z
end

zfromx(m::SVAE, x) = samplez(m, zparams(m, x)...)
