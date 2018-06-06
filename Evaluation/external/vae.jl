using Flux
using Adapt

import Flux: Chain, Dense

"""
		Implementation of VAE with Gaussian prior and posterior.

		VAE(q,g,β,s::Symbol = :unit) = VAE(q,g,β,Val(s))

		β is the weight on the KL-divergence
		s = :unit means that the output has unit variance
		s = :sigma means that the output learns the variance of Gaussians as well
"""
struct VAE{A,B,T<:AbstractFloat,V<:Val}
	q::A  # encoder (inference modul)
	g::B 	# decoder (generator)
	β::T 	#penalization
	variant::V
end

VAE(q,g,β,s::Symbol = :unit) = VAE(q,g,β,Val(s))

Flux.treelike(VAE)

kldiv(μ,σ2) = - mean(sum((@. log(σ2) - μ^2 - σ2), 1))
likelihood(x,μ) = - mean(sum((@. (x - μ)^2), 1))
likelihood(x,μ,σ2) = - mean(sum((@. (x - μ)^2/σ2 + log(σ2)), 1))/2

"""
		hsplitsoftp(x)

		split x horizontally into two equal parts and use softplus to the lower part

"""
hsplitsoftp(x) = x[1:size(x,1) ÷ 2,: ], softplus.(x[size(x,1) ÷ 2 + 1 : (size(x,1)) ,:])

function qg(m::VAE{A,B,T,V},x) where {A,B,T,V<:Val{:sigma}}
	μz, σ2z = hsplitsoftp(m.q(x))
	ϵ = randn(T,size(μz))
	μx, σ2x = hsplitsoftp(m.g(@. μz + √(σ2z)*ϵ))
	μz, σ2z, μx, σ2x
end

function loss(m::VAE{A,B,T,V},x) where {A,B,T,V<:Val{:sigma}}
	μz, σ2z, μx, σ2x = qg(m,x)
	-likelihood(x,μx,σ2x) + m.β*kldiv(μz,σ2z)
end

function loss(m::VAE{A,B,T,V},x) where {A,B,T,V<:Val{:unit}}
	μz, σ2z, μx = qg(m,x)
	-likelihood(x,μx) + m.β*kldiv(μz,σ2z)
end

function qg(m::VAE{A,B,T,V},x) where {A,B,T,V<:Val{:unit}}
	μz, σ2z = hsplitsoftp(m.q(x))
	ϵ = randn(T,size(μz))
	μx  = m.g(@. μz + √(σ2z)*ϵ)
	μz, σ2z, μx
end

function z(m::VAE{A,B,T,V},x) where {A,B,T,V}
	μz, σ2z = hsplitsoftp(m.q(x))
	ϵ = randn(T,size(μz))
	@. μz + √(σ2z)*ϵ
end
