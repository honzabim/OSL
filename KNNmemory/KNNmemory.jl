# NOT FINISHED!
mutable struct KNNmemory
    M::Array{Float64, 2}
    V::Array{Int64, 1}
    A::Array{Int64, 1}
    k::Integer
    α::Real

    function KNNmemory(memorySize::Integer, keySize::Integer, k::Integer, labelCount::Integer, α::Float64 = 0.1)
        M = rand(Float64, memorySize, keySize)
        V = rand(1:labelCount, memorySize)
        A = zeros(Int64, memorySize)

        for i = 1:memorySize
            M[i,:] = M[i,:] / norm(M[i,:])
        end

        new(M, V, A, k > memorySize ? memorySize : k, α)
    end
end

# Partitions the list l into chunks of the length n
partition(list, n) = [list[i:min(i + n - 1,length(list))] for i in 1:n:length(list)]


# This particular version of the query is not the most efficient, however, since it is only used for evaluation and
# not training it seemed reasonable
query(memory::KNNmemory, q::AbstractArray{Float64, 2}) = map(x -> query(memory, x), partition(q, size(q, 1)))

function query(memory::KNNmemory, q::AbstractArray{Float64, 1})
    similarity = memory.M * q
    largestID = indmax(similarity)
    return memory.V[largestID]
end

function findNearestPositiveAndNegative(memory::KNNmemory, kLargestIDs::Array{Int64, 1}, v::Integer)
    nearestPositiveID = nothing
    nearestNegativeID = nothing

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

memoryLoss(memory::KNNmemory, q::AbstractArray{Float64, 1}, nearestPosAndNegIDs::Tuple) = memoryLoss(memory, q, nearestPosAndNegIDs...)

function memoryLoss(memory::KNNmemory, q::AbstractArray{Float64, 1}, nearestPositiveID::Integer, nearestNegativeID::Integer)
    loss = max(dot(q, memory.M[nearestNegativeID, :]) - dot(q, memory.M[nearestPositiveID, :]) + memory.α, 0)
end

function memoryUpdate(memory::KNNmemory, q::AbstractArray{Float64, 1}, v::Integer, nearestNeighbourID::Integer)
    q = Flux.Tracker.data(q)
    if memory.V[nearestNeighbourID] == v
        memory.M[nearestNeighbourID, :] = (q + memory.M[nearestNeighbourID, :]) / norm((q + memory.M[nearestNeighbourID, :]))
        memory.A[nearestNeighbourID] = 0
    else
        oldestElementID = indmax(memory.A + rand(1:5))
        memory.M[oldestElementID, :] = q
        memory.V[oldestElementID] = v
        memory.A[oldestElementID] = 0
    end
end

# Update the age of all items
function increaseMemoryAge(memory::KNNmemory) 
    memory.A = memory.A + 1;
end

function trainQuery!(memory::KNNmemory, q::AbstractArray{Float64, 1}, v::Integer)
    # Find k nearest neighbours
    similarity = memory.M * q
    kLargestIDs = selectperm(similarity, 1:memory.k, rev = true)
    n1 = kLargestIDs[1]
    nearestNeighbour = memory.V[n1]

    loss = memoryLoss(memory, q, findNearestPositiveAndNegative(memory, kLargestIDs, v))
    memoryUpdate(memory, q, v, n1)
    increaseMemoryAge(memory)

    return loss
end

function trainQuery!(memory::KNNmemory, q::AbstractArray{Float64, 2}, v::Array{Int64, 1})
    # Find k nearest neighbours and compute losses
    batchSize = size(q, 2)
    similarity = memory.M * q
    loss::Flux.Tracker.TrackedReal{Float64} = 0.
    nearestNeighbourIDs = zeros(Integer, batchSize)

    for i in 1:batchSize
        kLargestIDs = selectperm(similarity[:, i], 1:memory.k, rev = true)
        nearestNeighbourIDs[i] = kLargestIDs[1];
        loss = loss + memoryLoss(memory, q[:, i], findNearestPositiveAndNegative(memory, kLargestIDs, v[i]))
    end

    # Memory update - cannot be done above because we have to compute all losses before changing the memory
    for i in 1:batchSize
        memoryUpdate(memory, q[:, i], v[i], nearestNeighbourIDs[i])
    end
    increaseMemoryAge(memory)

    return loss / batchSize
end
