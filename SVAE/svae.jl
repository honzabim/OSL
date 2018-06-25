include(bessel.jl)

using Flux
using NNlib
using Distributions

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

function zparams(model::SVAE, x)
	_zparams = model.q(x)
	return model.μzfromhidden(_zparams), model.κzfromhidden(_zparams)
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

function z(μz, κz)

end
