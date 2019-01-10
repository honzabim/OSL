using JLD2
using FileIO
using DataFrames
using Statistics
using CSV
using Crayons
using Crayons.Box
using StatsBase

const dataFolder = "d:/dev/julia/OSL/experiments/WSVAElongProbMem/"
const dataFolder = "d:/dev/julia/OSL/experiments/WSVAEshortProbMem/"
const knnfolder = "d:/dev/julia/data/knn/"
const datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "cardiotocography", "ecoli", "glass", "haberman", "ionosphere", "iris", "magic-telescope", "musk-2", "ionosphere", "page-blocks", "parkinsons", "pendigits", "pima-indians", "sonar", "spect-heart", "statlog-satimage", "statlog-vehicle", "synthetic-control-chart", "wall-following-robot", "waveform-1", "waveform-2", "wine", "yeast"]
const models = ["svae"]
const anomalycount = 10

loadExperiment(filePath) = load(filePath)["results"]


params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :memorysize, :k, :β, :anomaliesSeen, :aucpxv, :aucf2, :aucf3, :ar, :i, :κ, :anom_sel, :model, :dataset]
types = [Int, Int, Int, String, String, Int, Int, Float64, Int, Float64, Float64, Float64, Float64, Int, Float64, String, String, String]

const anomalycount = 10
function processFile!(dataframe, model, dataset)
    for i in 1:10
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
                    push!(dataframe, vcat(pars..., results[i][2][j][1:4]..., results[i][2][j][7:10]..., model, dataset))
                end
            end
        end
    end
end

allData = DataFrame(types, params, 0)
foreach((t) -> processFile!(allData, t[1], t[2]), Base.product(models, datasets))

CSV.write(dataFolder * "results.csv", allData)

function aggrmaxmean(df::DataFrame)
    dfagg = []
    for (ar, asel, d) in Base.product(unique(df[:ar]), unique(df[:anom_sel]), unique(df[:dataset]))
        dfaseen = df[(df[:dataset] .== d) .& (df[:anom_sel] .== asel) .& (df[:ar] .== ar), :]
        for aseen in unique(dfaseen[:anomaliesSeen])
            dfall = dfaseen[dfaseen[:anomaliesSeen] .== aseen, :]
            itermax = []
            for i in 1:maximum(dfall[:i])
                dfperi = dfall[dfall[:i] .== i, :]
                push!(itermax, [maximum(dfperi[:aucpxv]), maximum(dfperi[:aucf2]), maximum(dfperi[:aucf3])])
            end
            itermax = hcat(itermax...)
            meanvals = mean(itermax, dims = 2)
            push!(dfagg, DataFrame(dataset = d, anom_sel = asel, anom_ratio = ar, anom_seen = aseen, pxvita = meanvals[1], f2 = meanvals[2], f3 = meanvals[3]))
        end
    end
    return vcat(dfagg...)
end

function aggrmeanmax(df::DataFrame)
    dfagg = []
    for (ar, asel, d) in Base.product(unique(df[:ar]), unique(df[:anom_sel]), unique(df[:dataset]))
        dfaseen = df[(df[:dataset] .== d) .& (df[:anom_sel] .== asel) .& (df[:ar] .== ar), :]
        for aseen in unique(dfaseen[:anomaliesSeen])
            dfall = dfaseen[dfaseen[:anomaliesSeen] .== aseen, :]
            parmean = []
            for (ms, b, k) in Base.product(unique(df[:memorysize]), unique(df[:β]), unique(df[:κ]))
                dfperpars = dfall[(dfall[:memorysize] .== ms) .& (dfall[:β] .== b) .& (dfall[:κ] .== k), :]
                push!(parmean, [mean(dfperpars[:aucpxv]), mean(dfperpars[:aucf2]), mean(dfperpars[:aucf3])])
            end
            parmean = hcat(parmean...)
            maxvals = maximum(parmean, dims = 2)
            push!(dfagg, DataFrame(dataset = d, anom_sel = asel, anom_ratio = ar, anom_seen = aseen, pxvita = maxvals[1], f2 = maxvals[2], f3 = maxvals[3]))
        end
    end
    return vcat(dfagg...)
end

function cmpnonsampledz(nzdf, szdf)
    df = []
    for (ar, asel, d) in Base.product(unique(nzdf[:anom_ratio]), unique(nzdf[:anom_sel]), unique(nzdf[:dataset]))
        nzdfaseen = nzdf[(nzdf[:dataset] .== d) .& (nzdf[:anom_sel] .== asel) .& (nzdf[:anom_ratio] .== ar), :]
        szdfaseen = szdf[(szdf[:dataset] .== d) .& (szdf[:anom_sel] .== asel) .& (szdf[:anom_ratio] .== ar), :]
        for aseen in unique(nzdfaseen[:anom_seen])
            nz = nzdfaseen[nzdfaseen[:anom_seen] .== aseen, :]
            sz = szdfaseen[szdfaseen[:anom_seen] .== aseen, :]
            push!(df, DataFrame(dataset = d, anom_sel = asel, anom_ratio = ar, anomaliesSeen = aseen, nzpxv = nz[:pxvita], szpxv = sz[:pxvita], nzf2 = nz[:f2], szf2 = sz[:f2], nzf3 = nz[:f3], szf3 = sz[:f3]))
        end
    end
    return vcat(df...)
end

function printbest3(df, cmp_name)
    collen = 20
    maxlen = [maximum(vcat(length.(df[:dataset])..., 45))]
    maxlen = vcat(maxlen, 10, 10, 10)
    names = ["dataset", "anom_sel", "anom_ratio", "anom_seen"]
    metds = [cmp_name, "f2", "f3"]

    for i in 1:4
        print(names[i])
        print(repeat(" ", maxlen[i] - length(names[i])) * " | ")
    end
    for i in 1:3
        print(metds[i])
        print(repeat(" ", collen - length(metds[i])) * " | ")
    end
    println()
    crays = [Crayon(foreground = :red), Crayon(foreground = :yellow), Crayon(foreground = :green)]
    defc = Crayon(reset = true)
    for i in 1:size(df, 1)
        for j in 1:4
            print(defc, df[i, j])
            print(defc, repeat(" ", maxlen[j] - length("$(df[i, j])")) * " | ")
        end
        aucs = df[i, 5:7]
        p = ordinalrank(vec(convert(Array, aucs)))
        for c in 1:3
            s = "$(aucs[c])"
            print(crays[p[c]], s)
            print(defc, repeat(" ", collen - length(s)) * " | ")
        end
        println()
    end
end

function loadknn()
    df = []
    for d in datasets
        filename = knnfolder * d * "/knn_easy_0.05_low.csv"
        if !isfile(filename)
            println("$filename not found.")
        else
            ddf = CSV.read(filename)
            pardf = []
            for (prn, k) in Base.product(unique(ddf[:prname]), unique(ddf[:k]))
                seldf = ddf[(ddf[:prname] .== prn) .& (ddf[:k] .== k), :]
                push!(pardf, DataFrame(dataset = d, prname = prn, k = k, auc = mean(seldf[:auc_test])))
            end
            push!(df, vcat(pardf...))
        end
    end
    df = vcat(df...)
end

function comparewithknn(knndf, svaedf)
    df = []
    for d in unique(svaedf[:dataset])
        knnauc = maximum(knndf[knndf[:dataset] .== d, :][:auc])
        svaeddf = svaedf[svaedf[:dataset] .== d, :]
        svaeddf[:knnauc] = knnauc
        push!(df, svaeddf)
    end
    return vcat(df...)
end
