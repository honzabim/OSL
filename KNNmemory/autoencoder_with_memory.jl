using Flux

push!(LOAD_PATH, pwd())
using KNNmemory


mutable struct AutoencoderWithMemory
    encoder
    decoder
    memory::KNNmemory

    function AutoencoderWithMemory(encoder, decoder, )
end
