using Flux

push!(LOAD_PATH, pwd(), "/home/jan/dev/anomaly detection/anomaly_detection/src", "/home/jan/dev/FluxExtensions.jl/src")
using KNNmem
using AnomalyDetection

dataPath = "/home/jan/dev/data/loda/public/datasets/numerical"
data = AnomalyDetection.loaddata(dataPath)
