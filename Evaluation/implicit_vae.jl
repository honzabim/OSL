using Flux

S(μ, σ, z) = @. (z - μ) / σ
Sinv(μ, σ, ϵ) = @. μ +  σ * ϵ
Sinv(μ::TrackedArray, σ::TrackedArray, ϵ) = Tracker.track(Sinv, μ, σ, ϵ)
Sinv(μ::Flux.Tracker.TrackedReal, σ::Flux.Tracker.TrackedReal, ϵ) = Tracker.track(Sinv, μ, σ, ϵ)

function Tracker.back(::typeof(Sinv), Δ, μ, σ, ϵ)
    _z = Sinv(Flux.Tracker.data(μ), Flux.Tracker.data(σ), Flux.Tracker.data(ϵ))

    _μ = Flux.Tracker.data(μ)
    _σ = Flux.Tracker.data(σ)

    _μ = param(_μ)
    _σ = param(_σ)
    _z = param(_z)

    l = S(_μ, _σ, _z)
    Flux.Tracker.back!(l)

    dSdz = Flux.Tracker.grad(_z)
    dSdμ = Flux.Tracker.grad(_μ)
    dSdσ = Flux.Tracker.grad(_σ)

    dzdμ = -dSdμ ./ dSdz
    dzdσ = -dSdσ ./ dSdz

    (Tracker.@back(μ, dzdμ .* Δ); Tracker.@back(σ, dzdσ .* Δ))
end

mu = 0
sigma = 2

mu = param(mu)
sigma = param(sigma)

e = randn()

z = Sinv(mu, sigma, e)

Flux.Tracker.back!(z)

Flux.Tracker.grad(mu)
Flux.Tracker.grad(sigma)

S⁻¹
