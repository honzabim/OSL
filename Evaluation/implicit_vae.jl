using Flux

S(z, μ, σ ) = @. (z - μ) / σ
Sinv(ϵ, μ, σ) = @. μ +  σ * ϵ
Sinv(ϵ, μ::TrackedArray, σ::TrackedArray) = Tracker.track(Sinv, ϵ, μ, σ)
Sinv(ϵ, μ::Flux.Tracker.TrackedReal, σ::Flux.Tracker.TrackedReal) = Tracker.track(Sinv, ϵ, μ, σ)

function ∇Sinv(ϵ, params...)
    _params = map(x -> Flux.Tracker.data(x), params)
    _z = param(Sinv(Flux.Tracker.data(ϵ), _params...))
    _params = map(x -> param(x), _params)

    l = S(_z, _params...)
    Flux.Tracker.back!(l)

    dSdz = Flux.Tracker.grad(_z)
    dSdϕ = map(x -> Flux.Tracker.grad(x), _params)

    dzdϕ = map(x -> -x ./ dSdz, dSdϕ)

    # foreach((x, dx) -> Tracker.@back(x, dx .* Δ), params, dzdϕ)
end

function Tracker.back(::typeof(Sinv), Δ, ϵ, μ, σ)
    _z = Sinv(Flux.Tracker.data(ϵ), Flux.Tracker.data(μ), Flux.Tracker.data(σ))

    _μ = Flux.Tracker.data(μ)
    _σ = Flux.Tracker.data(σ)

    _μ = param(_μ)
    _σ = param(_σ)
    _z = param(_z)

    l = S(_z, _μ, _σ)
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
