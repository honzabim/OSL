using JLD2
using FileIO
using DataFrames
using Statistics
using CSV
using Crayons
using Crayons.Box
using StatsBase

const dataFolder = "d:/dev/julia/OSL/experiments/WSVAElongSVAEanom/"
const knnfolder = "d:/dev/julia/data/knn/"
const datasets = ["abalone", "blood-transfusion", "breast-tissue", "haberman", "iris", "breast-cancer-wisconsin", "cardiotocography", "ecoli", "glass", "pima-indians", "sonar", "statlog-satimage", "waveform-1", "waveform-2", "yeast", "ionosphere", "wine"]
const models = ["svae"]
const anomalycount = 5

loadExperiment(filePath) = load(filePath)["results"]


params = [:hidden, :latent, :layers, :nonlinearity, :layertype, :β, :anomaliesSeen, :auc, :aucpxv, :ar, :i, :α, :model, :dataset]
types = [Int, Int, Int, String, String, Float64, Int, Float64, Float64, Float64, Int, Float64, String, String]

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
                    pars = length(results[1][1]) > 5 ? vcat(results[i][1][1:6]...) : vcat(results[i][1]..., -1, -1)
                    # push!(dataframe, vcat(pars..., results[i][2][ac][1:3]..., model, dataset))
                    push!(dataframe, vcat(pars..., results[i][2][j][1:3]..., results[i][2][j][5:7]..., model, dataset))
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
        for aseen in unique(dfaseen[:anomaliesSeen])
            dfall = dfaseen[dfaseen[:anomaliesSeen] .== aseen, :]
            parmean = []
            for (a, b) in Base.product(unique(df[:α]), unique(df[:β]))
                dfperpars = dfall[(dfall[:α] .== a) .& (dfall[:β] .== b), :]
                push!(parmean, [mean(dfperpars[:aucpxv]), mean(dfperpars[:auc])])
            end
            parmean = hcat(parmean...)
            maxvals = maximum(parmean, dims = 2)
            push!(dfagg, DataFrame(dataset = d, anom_ratio = ar, anom_seen = aseen, pxvita = maxvals[1], auc = maxvals[2]))
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

function printbest3(df)
    collen = 20
    maxlen = [maximum(vcat(length.(df[:dataset])..., 45))]
    maxlen = vcat(maxlen, 10, 10, 10)
    names = ["dataset", "anom_ratio", "anom_seen"]
    metds = ["pxvita", "auc", "knnauc"]

    for i in 1:3
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
        for j in 1:3
            print(defc, df[i, j])
            print(defc, repeat(" ", maxlen[j] - length("$(df[i, j])")) * " | ")
        end
        aucs = df[i, 4:6]
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
            pardf = vcat(pardf...)
            pardf[:anom_ratio] = 0.05
            push!(df, pardf)
        end
        for ar in [0.01, 0.005]
            filename = knnfolder * d * "/knn_easy_$(ar)_low.jld2"
            if !isfile(filename)
                println("$filename not found.")
            else
                ddf = load(filename)["auc"]
                pardf = []
                for (prn, k) in Base.product(unique(ddf[:prname]), unique(ddf[:k]))
                    seldf = ddf[(ddf[:prname] .== prn) .& (ddf[:k] .== k), :]
                    push!(pardf, DataFrame(dataset = d, prname = prn, k = k, auc = mean(seldf[:auc_test])))
                end
                pardf = vcat(pardf...)
                pardf[:anom_ratio] = ar
                push!(df, pardf)
            end
        end
    end
    df = vcat(df...)
end

function selectminmaxanoms(df)
    ddf = []
    for (ar, d) in Base.product(unique(df[:anom_ratio]), unique(df[:dataset]))
        seldf = df[(df[:dataset] .== d) .& (df[:anom_ratio] .== ar), :]
        maxas = maximum(seldf[:anom_seen])
        push!(ddf, seldf[seldf[:anom_seen] .== 1, :])
        if (maxas != 1)
            push!(ddf, seldf[seldf[:anom_seen] .== maxas, :])
        end
    end
    return vcat(ddf...)
end

function comparewithknn(knndf, svaedf)
    df = []
    for (ar, d) in Base.product(unique(svaedf[:anom_ratio]), unique(svaedf[:dataset]))
        knnauc = maximum(knndf[(knndf[:dataset] .== d) .& (knndf[:anom_ratio] .== ar), :][:auc])
        svaeddf = svaedf[(svaedf[:dataset] .== d) .& (svaedf[:anom_ratio] .== ar), :]
        svaeddf[:knnauc] = knnauc
        push!(df, svaeddf)
    end
    return vcat(df...)
end
