include("bessel.jl")

using Flux
using NNlib
using Distributions
using SpecialFunctions

struct SVAE
	q	# encoder (inference modul)
	g	# decoder (generator)
	zdim
	μzfromhidden # function to compute μz from the hidden layer
	κzfromhidden # function to compute κz from the hidden layer

	'''
	SVAE(q, g, hdim, zdim) Constructor of the S-VAE with hidden dim `hdim` and latent dim = `zdim`. `zdim > 3`
	'''
	SVAE(q, g, hdim::Integer, zdim::Integer) = SVAE(q, g, zdim, Dense(hdim, zdim, normalize), Dense(hdim, 1, softplus))
end

function loss(m::SVAE, x)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + kldiv(m, κz)
end

function zparams(model::SVAE, x)
	_zparams = model.q(x)
	return model.μzfromhidden(_zparams), model.κzfromhidden(_zparams)
end

function kldiv(model::SVAE, κ)
	m = model.zdim
	return κ * besselix(m / 2, κ) / besselix(m / 2 - 1, κ) + (m / 2 - 1) * log(κ) - (m / 2) * log(2π) -
			(κ log(besselix(m / 2 - 1, κ))) + m / 2 * log(π) + log(2) - lgamma(m / 2)
end

function sampleω(model::SVAE, κ)
	m = model.zdim
	c = √(4κ ^ 2 + (m - 1) ^ 2)
	b = (-2κ + c) / (m - 1)
	a = (m - 1 + 2κ + c) / 4
	d = (4 * a * b) / (1 + b) - (m - 1) * log(m - 1)

	uniform = Uniform()
	beta = Beta((m - 1) / 2., (m - 1) / 2.)

	while true
		ϵ = rand(beta)
		ω = (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
		t = 2 * a * b / (1 - (1 - b) * ϵ)
		u = rand(uniform)

		if ((m - 1) * log(t) - t + d) >= log(u)
			break
		end
	end
	return ω
end

function householderrotation(x, μ)
	e1 = zeros(μ)
	e1[1] = 1
	u = normalize(e1 - μ)
	z = x' - 2 * u * u' * x'
end

function samplez(m::SVAE, μz, κz)
	ω = sampleω(m, κz)

	normal = Normal()
	v = rand(normal, length(μz) - 1)
	z = householderrotation(vcat(ω, √(1 - ω ^ 2) * v), μz)
	return z
end

zfromx(m::SVAE, x) = samplez(m, zparams(m, x)...)
