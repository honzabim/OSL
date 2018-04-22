using Flux

push!(LOAD_PATH, pwd())
using KNNmemory

const DEFAULT_LABEL = 0

mutable struct AutoencoderWithMemory
    encoder
    decoder
    memory::KNNmemory

    function AutoencoderWithMemory(encoder, decoder, memorySize::Integer, keySize::Integer, k::Integer, labelCount::Integer, α::Float64 = 0.1)
        return new(encoder, decoder, KNNmemory(memorySize, keySize, k, labelCount, α))
    end
end

reconstructionError(ae::AutoencoderWithMemory, x) = Flux.mse(ae.decoder(ae.encoder(x)), x)
memoryTrainQuery(ae::AutoencoderWithMemory, data, label) = trainQuery!(ae.memory, ae.encoder(data), label)
learnAnomaly(ae::AutoencoderWithMemory, data, label, β) = reconstructionError(ae, data) + β * trainQuery!(ae.memory, ae.encoder(data), label)
memoryClassify(ae::AutoencoderWithMemory, x) = query(ae, ae.encoder(x))

function learnRepresentation(ae::AutoencoderWithMemory, x)
    latentVariable = ae.encoder(x)
    trainQuery!(ae.memory, latentVariable, DEFAULT_LABEL)
    return Flux.mse(ae.decoder(latentVariable))
end


