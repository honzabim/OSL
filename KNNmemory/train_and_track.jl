using Flux
using ValueHistories

function trainAndTrack!(loss, optimizer, iterations::Integer, batchSize::Integer, trainData::AbstractArray, trainLabels::AbstractArray, testData::AbstractArray, testLabels::AbstractArray, printInterationCount = 100)
    history = MVHistory()
    storedLoss = 0.

    for i in 1:iterations
        # sample from data
        (x, y) = sample(batchSize, trainData, trainLabels)

        # gradient computation and update
        l = loss(x, y)
        storedLoss += Flux.Tracker.data(l)

        # print to console
        if i % printInterationCount == 0
            println(Flux.Tracker.data(l))

            # store to memory
            push!(history, :train, i, storedLoss / printInterationCount)
            push!(history, :test, i, Flux.Tracker.data(loss(testData, testLabels)))
            storedLoss = 0.
        end
        Flux.Tracker.back!(l)
        optimizer()
    end

    return history
end

function sample(batchSize::Integer, data::AbstractArray{T, 2} where T, labels::AbstractArray{T, 1} where T)
    batchIndeces = rand(1:size(data, 2), batchSize)
    x = data[:, batchIndeces];
    y = labels[batchIndeces];
    return x, y
end

function sample(batchSize::Integer, data::AbstractArray{T, 2} where T, labels::AbstractArray{T, 2} where T)
    batchIndeces = rand(1:size(data, 2), batchSize)
    x = data[:, batchIndeces];
    y = labels[:, batchIndeces];
    return x, y
end
