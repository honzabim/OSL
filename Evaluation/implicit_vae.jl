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
    return dzdϕ
end

# It does not work with (::typeof(Sinv), Δ, ϵ, params...) for some reason...
function Tracker.back(::typeof(Sinv), Δ, ϵ, μ, σ)
    params = (μ, σ)
    dzdϕ = ∇Sinv(ϵ, params...)
    foreach((x, dx) -> Tracker.@back(x, dx .* Δ), params, dzdϕ)
end

mu = 0
sigma = 2

mu = param(mu)
sigma = param(sigma)

e = randn()

z = Sinv(e, mu, sigma)

Flux.Tracker.back!(z)

Flux.Tracker.grad(mu)
Flux.Tracker.grad(sigma)
