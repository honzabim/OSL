using JLD2
using FileIO
using DataFrames
using Statistics
using CSV
using Crayons
using Crayons.Box
using StatsBase
using Plots

# const dataFolder = "d:/dev/julia/OSL/experiments/WSVAEvsTDlong/"
const dataFolder = "d:/dev/julia/OSL/experiments/WSVAEvsTDnewacqAUC/"
const tdfolder = "d:/dev/experiments/FeedbackIsolationForest/test/svaetest/out/"
const datasets = ["abalone", "blood-transfusion", "breast-tissue", "ecoli", "glass", "haberman", "ionosphere", "iris", "pima-indians", "statlog-satimage", "waveform-1", "waveform-2", "wine", "yeast"]
const models = ["svae"]

loadExperiment(filePath) = load(filePath)["results"]

params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :β, :fc, :aucpxv, :aucf2, :aucf3, :ar, :i, :κ, :anom_disc, :model, :dataset]
types = [Int, Int, Int, String, String, Int, Int, Float64, Int, Float64, Float64, Float64, Float64, Int, Float64, Int, String, String]

function processFile!(dataframe, model, dataset)
    for i in 1:5
        println("Processing $model $dataset $i")
        filename = dataFolder * dataset * "-$i-" * model * ".jld2"
        if !isfile(filename)
            println("$filename not found.")
        else
            results = loadExperiment(dataFolder * dataset * "-$i-" * model * ".jld2")

            for i in 1:length(results)
                for j in 1:length(results[1][2])
                    pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:7]..., results[i][1][9]) : vcat(results[i][1]..., -1, -1)
                    # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                    push!(dataframe, vcat(pars..., results[i][2][j][1:8]..., model, dataset))
                end
            end
        end
    end
end

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))

CSV.write(dataFolder * "results.csv", allData)

function aggrmeanmax(df::DataFrame)
    dfagg = []
    for (ar, d) in Base.product(unique(df[:ar]), unique(df[:dataset]))
        dfaseen = df[(df[:dataset] .== d) .& (df[:ar] .== ar), :]
        for fc in 1:50
            parmean = []
            for (ms, b, k) in Base.product(unique(df[:memorysize]), unique(df[:β]), unique(df[:κ]))
                dfperpars = dfaseen[(dfaseen[:memorysize] .== ms) .& (dfaseen[:β] .== b) .& (dfaseen[:κ] .== k) .& (dfaseen[:fc] .== fc), :]
                push!(parmean, [mean(dfperpars[:anom_disc]), mean(dfperpars[:aucf3])])
            end
            parmean = hcat(parmean...)
            maxad = maximum(parmean, dims = 2)
            push!(dfagg, DataFrame(dataset = d, anom_ratio = ar, fc = fc, auc = maxad[1], anom_discovered = maxad[2]))
        end
    end
    return vcat(dfagg...)
end

function loadtd(df)
    newdf = []
    for (ar, d) in Base.product(unique(df[:anom_ratio]), unique(df[:dataset]))
        seldf = df[(df[:anom_ratio] .== ar) .& (df[:dataset] .== d), :]
        vals = []
        for i in 1:5
            filename = tdfolder * "$d-$ar-$i" * "_summary_feed_50_losstype_logistic_updatetype_online_ngrad_1_reg_0_lrate_1_pwgt_0_inwgt_0_rtype_L2.csv"
            if !isfile(filename)
                println("$filename not found.")
            else
                tddf = CSV.read(filename)
                ads = tddf[:, length(names(tddf))]
                vals = vcat(vals, ads...)
            end
        end
        seldf[:anom_discovered_td] = mean(vals)
        push!(newdf, seldf)
    end
    return vcat(newdf...)
end

cmpdf = CSV.read(dataFolder * "cmp-results.csv")

maxstr = collect(0.1:0.016:0.9)[1:50]
clsstr = collect(0.2:0.012:0.8)[1:50]

plotlyjs()
p = plot(maxstr, seriescolor = "#1B9CE5", xlabel = "# feedback iterations", guidefont=font(10), tickfont=font(9), legendfont=font(10), grid = false, ylabel = "AUC", label = "max strategy", legend = :top, size = (400, 200))
p = plot!(clsstr, seriescolor = "#F51069", label = "cluster strategy")
savefig(p, "figures/al_strat_cmp.pdf")
