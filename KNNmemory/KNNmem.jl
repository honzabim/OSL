"""
This module implements a memory, according to https://arxiv.org/abs/1703.03129,
that can be used as the last layer in a NN in the Flux framework.
"""
module KNNmem

using Flux
using Juno
using Random
using LinearAlgebra
using SpecialFunctions

export KNNmemory, query, trainQuery!, augmentModelWithMemory

"""
    KNNmemory{T}
Structure that contains all the memory data.
"""
mutable struct KNNmemory{T <: Real}
    M::Matrix{T} # keys in the memory
    V::Vector{<:Integer} # values in the memory (labels)
    A::Vector{<:Integer} # age of a given key-value pair
    k::Integer # number of neighbors used in kNN
    α::Real # parameter setting the required distance between the nearest positive and negative sample in the kNN

    """
        KNNmemory{T}(memorySize, keySize, k, labelCount, [α])
    Memory constructor that initializes it with random keys and random labels.
    # Arguments
    - `memorySize::Integer`: number of keys that can be stored in the memoryUpdate!
    - `keySize::Integer`: length of a key
    - `k::Integer`: number of k nearest neighbors to look for
    - `labelCount::Integer`: number of labels that are in the dataset (used for the random initialization)
    - `α::Real`: parameter of the memory loss function that determines required distance between clusters
    """
    function KNNmemory{T}(memorySize::Integer, keySize::Integer, k::Integer, labelCount::Integer, α::Real = 0.1) where T
        M = rand(T, memorySize, keySize) .* 2 .- 1
        V = rand(0:(labelCount - 1), memorySize)
        A = zeros(Int, memorySize)

        for i = 1:memorySize
            M[i,:] = normalize(M[i,:])
        end

        new(M, V, A, k > memorySize ? memorySize : k, convert(T, α))
    end
end

"""
    findNearestPositiveAndNegative(memory, kLargestIDs, v)
For given set of k nearest neighbours, find the closest two that have the same label `v` and a different label respectively.
`kLargestIDs::Vector{<:Integer}` contains k indices leading to the k most similar keys in the memorySize.
"""
function findNearestPositiveAndNegative(memory::KNNmemory, kLargestIDs::Vector{<:Integer}, v::Integer)
    nearestPositiveID = nothing
    nearestNegativeID = nothing

    # typically this should not result into too many iterations
    for i in 1:memory.k
        if nearestPositiveID == nothing && memory.V[kLargestIDs[i]] == v
            nearestPositiveID = kLargestIDs[i]
        end
        if nearestNegativeID == nothing && memory.V[kLargestIDs[i]] != v
            nearestNegativeID = kLargestIDs[i]
        end
        if nearestPositiveID != nothing && nearestNegativeID != nothing
            break
        end
    end

    #= We assume that there exists such i that memory.V[i] == v
        and also such j that memory.V[j] != v
list
        We also assume that this won't happen very often, otherwise,
        we would need to randomize this selection (possible TODO) =#

    if nearestPositiveID == nothing
        nearestPositiveID = indmax(memory.V .== v)
    end
    if nearestNegativeID == nothing
        nearestNegativeID = indmax(memory.V .!= v)
    end

    return nearestPositiveID, nearestNegativeID
end

"""
    Base.normalize(v::Flux.Tracker.TrackedArray{T})
Reimplementation of the inbuild function normalize() so that it works with tracked (Flux.Tracker) vectors.
"""
Base.normalize(v::Flux.Tracker.TrackedArray{T}) where {T <: Real} = v ./ (sqrt(sum(v .^ 2) + eps(T)))

"""
    memoryLoss(memory, q, nearestPosAndNegIDs)
Loss generated by the memory based on the lookup of a key-value pair for the key `q` - exactly as in the paper.
`nearestPosAndNegIDs::Tuple` represents ids of the closest key with the same and a different label respectively.
"""
memoryLoss(memory::KNNmemory{T}, q::AbstractArray{T, 1}, nearestPosAndNegIDs::Tuple) where {T} = memoryLoss(memory, q, nearestPosAndNegIDs...)

function memoryLoss(memory::KNNmemory{T}, q::AbstractArray{T, 1}, nearestPositiveID::Integer, nearestNegativeID::Integer) where {T}
    loss = max(dot(normalize(q), memory.M[nearestNegativeID, :]) - dot(normalize(q), memory.M[nearestPositiveID, :]) + memory.α, 0)
end

"""
    normalizeQuery(q)
Normalizes all query keys in the matrix separately by converting it into an array of arrays.
"""
normalizeQuery(q) = hcat((normalize.([q[:, i] for i in 1:size(q, 2)]))...)
normalizecolumns(m) = m ./ sqrt.(sum(m .^ 2, dims = 1) .+ eps(eltype(Flux.Tracker.data(m))))

"""
    memoryUpdate!(memory, q, v, nearestNeighbourID)
It computes the appropriate update of the memory after a key-value pair was lookedup in it for the key `q` and expected label `v`.
"""
function memoryUpdate!(memory::KNNmemory{T}, q::Vector{T}, v::Integer, nearestNeighbourID::Integer) where {T}
    # If the memory return the correct value for the given key, update the centroid
    if memory.V[nearestNeighbourID] != 1 && memory.V[nearestNeighbourID] == v # TODO: This is a hack to not move anomalies - should be done in a better way!
        memory.M[nearestNeighbourID, :] = normalize(q + memory.M[nearestNeighbourID, :])
        memory.A[nearestNeighbourID] = 0

    # If the memory did not return the correct value for the given key, store the key-value pair instead of the oldest element
    else
        oldestElementID = indmax(memory.A + rand(1:5))
        memory.M[oldestElementID, :] = q
        memory.V[oldestElementID] = v
        memory.A[oldestElementID] = 0
    end
end

"""
    increaseMemoryAge(memory)
Update the age of all items
"""
function increaseMemoryAge(memory::KNNmemory)
    memory.A += 1;
end

"""
    query(memory, q)
Returns the nearest neighbour's value and its confidence level for a given key `q` but does not modify the memory itself.
"""
function query(memory::KNNmemory{T}, q::AbstractArray{T, N} where N) where {T}
    similarity = memory.M * normalizeQuery(Flux.Tracker.data(q))
    values = memory.V[Flux.argmax(similarity)]

    function probScorePerQ(index)
        kLargestIDs = collect(partialsortperm(similarity[:, index], 1:memory.k, rev = true))
        probsOfNearestKeys = softmax(similarity[kLargestIDs, index])
        nearestValues = memory.V[kLargestIDs]
        return sum(probsOfNearestKeys[nearestValues .== 1])
    end

    return values, map(probScorePerQ, 1:size(q, 2)) # basicaly returns the nearest value in the memory + sum of probs of anomalies that are in the k-nearest
end

# logc(p, κ) = (p / 2 - 1) * log(κ) - (p / 2) * log(2π) - log(besseli(p / 2 - 1, κ))
logc(p, κ) = (p ./ 2 .- 1) .* log.(κ) .- (p ./ 2) .* log(2π) .- κ .- log.(besselix(p / 2 - 1, κ))


# log likelihood of one sample under the VMF dist with given parameters
# log_vmf(x, μ, κ) = κ * μ' * x .+ log.(c(length(μ), κ))
log_vmf(x, μ, κ) = κ * μ' * x .+ logc(length(μ), κ)

vmf_mix_lkh(x, μs) = vmf_mix_lkh(x, μs, size(μs, 2))
function vmf_mix_lkh(x, μs, μlength::Integer)
    κs = ones(μlength) .* 3 # This is quite arbitrary as we don't really know what to use for kappa but it shouldn't matter if it is the same
	l = 0
	for K in 1:μlength
		l += exp.(log_vmf(x, μs[:, K], κs[K]))
	end
	l /= μlength
	return l
end

"""
    prob_query(memory, q)
Returns the nearest neighbour's value and its confidence level for a given key `q` but does not modify the memory itself.
"""
function prob_query(memory::KNNmemory{T}, q::AbstractArray{T, N} where N) where {T}
    normq = normalizecolumns(Flux.Tracker.data(q))
    similarity = memory.M * normq
    values = memory.V[Flux.argmax(similarity)]

    function probScorePerQ(index)
        kLargestIDs = collect(partialsortperm(similarity[:, index], 1:memory.k, rev = true))
		nearestAnoms = collect(memory.M[kLargestIDs, :][memory.V[kLargestIDs] .== 1, :]')
		if length(nearestAnoms) == 0
			return 0
		elseif size(nearestAnoms, 2) == memory.k
			return 1
		end
		nearestNormal = collect(memory.M[kLargestIDs, :][memory.V[kLargestIDs] .== 0, :]')
		pxgivena = vmf_mix_lkh(normq[:, index], nearestAnoms)
		pxgivenn = vmf_mix_lkh(normq[:, index], nearestNormal)
		return pxgivena / (pxgivena + pxgivenn)
    end

    return values, map(probScorePerQ, 1:size(q, 2)) # basicaly returns the nearest value in the memory + sum of probs of anomalies that are in the k-nearest
end



"""
    trainQuery!(memory, q, v)
Query 'q' to the memory that does update its content and returns a loss for expected outcome label `v`.
"""
trainQuery!(memory::KNNmemory{T}, q::AbstractArray{T, N} where N, v::Integer) where {T} = trainQuery!(memory, q, [v])

function trainQuery!(memory::KNNmemory{T}, q::AbstractArray{T, N} where N, v::Vector{<:Integer}) where {T}
    # Find k nearest neighbours and compute losses
    batchSize = size(q, 2)
    normalizedQuery = normalizeQuery(Flux.Tracker.data(q))
    similarity = memory.M * normalizedQuery # computes all similarities of all qs and all keys in the memory at once
    loss::Flux.Tracker.TrackedReal{T} = 0 # loss must be tracked; otherwise flux cannot use it
    nearestNeighbourIDs = zeros(Integer, batchSize)

    for i in 1:batchSize
        kLargestIDs = collect(partialsortperm(similarity[:, i], 1:memory.k, rev = true))
        nearestNeighbourIDs[i] = kLargestIDs[1];
        loss += memoryLoss(memory, q[:, i], findNearestPositiveAndNegative(memory, kLargestIDs, v[i]))
    end

    # Memory update - cannot be done above because we have to compute all losses before changing the memory
    for i in 1:batchSize
        memoryUpdate!(memory, normalizedQuery[:, i], v[i], nearestNeighbourIDs[i])
    end
    increaseMemoryAge(memory)

    return loss / batchSize
end

# function augmentModelWithMemoryProb(model, memorySize, keySize, k, labelCount, α = 0.1, T = Float32)
#     memory = KNNmemory{T}(memorySize, keySize, k, labelCount, α)
#     trainQ!(data, labels) = trainQuery!(memory, model(data), labels)
#     trainQOnLatent!(latentData, labels) = trainQuery!(memory, latentData, labels)
#     testQ(data) = prob_query(memory, model(data))
#     return trainQ!, testQ, trainQOnLatent!
# end


"""
    augmentModelWithMemory(model, memorySize, keySize, k, labelCount, [α = 0.1,] [T = Float32])
Creates a set of functions that allow for training and testing of a model whose outputs are used as keys to the memory.
"""
function augmentModelWithMemory(model, memorySize, keySize, k, labelCount, α = 0.1, T = Float32)
    memory = KNNmemory{T}(memorySize, keySize, k, labelCount, α)
    trainQ!(data, labels) = trainQuery!(memory, model(data), labels)
    trainQOnLatent!(latentData, labels) = trainQuery!(memory, latentData, labels)
    testQ(data) = prob_query(memory, model(data)) # TODO: this should be just query!
    return trainQ!, testQ, trainQOnLatent!
end
end
