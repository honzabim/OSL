using CSV
using DataFrames
using PaperUtils

dataFolder = "d:/dev/julia/OSL/experiments/WSVAElongProbMem/"
memdf = CSV.read(dataFolder * "pamet-svae_knn_selected_comparison.csv")

dataFolder = "d:/dev/julia/OSL/experiments/WSVAElongSVAEanom/"
twocapsdf = CSV.read(dataFolder * "polokoule-knn-comparison-selected-anoms.csv")

dataFolder = "d:/dev/julia/OSL/experiments/WSVAE_PXV_vs_PZ/"
pzpxdf = CSV.read(dataFolder * "agg-results.csv")

datasets = unique(memdf[:dataset])
anom_ratios = [0.05, 0.01, 0.005]

mergeddf = []

for (ar, d) in Base.product(anom_ratios, datasets)
    memsel = memdf[(memdf[:dataset] .== d) .& (memdf[:anom_ratio] .== ar), :]
    pxvita = memsel[:pxvita]
    knn = memsel[:knnauc]
    memauc = memsel[:f3]

    twocapssel = twocapsdf[(twocapsdf[:dataset] .== d) .& (twocapsdf[:anom_ratio] .== ar), :]
    tcauc = 0
    if size(twocapssel, 1) > 0
        tcauc = twocapssel[:auc]
    end

    pzpxsel = pzpxdf[(pzpxdf[:dataset] .== d) .& (pzpxdf[:anom_ratio] .== ar), :]
    pzauc = 0
    pxauc = 0
    if size(pzpxsel, 1) > 0
        pzauc = pzpxsel[:aucpz][1]
        pxauc = pzpxsel[:aucpxv][1]
    end
    if size(twocapssel, 1) > 1
        pzauc = [pzauc, pzauc]
        pxauc = [pxauc, pxauc]
    end

    ds = d
    anr = ar
    as = "1"

    ds = [d, d]
    anr = [ar, ar]
    as = ["1", "max"]

    # println("$ds, $anr, $as, $knn, $pxvita, $pzauc, $tcauc, $memauc")
    topercents = 100
    push!(mergeddf, DataFrame(dataset = ds[1], anom_ratio = anr[1], anom_seen = as[1], knn = knn[1] .* topercents, svae_px = pxvita[1] .* topercents, svae_pz = pzauc[1] .* topercents, two_caps_1 = tcauc[1] .* topercents, two_caps_max = length(tcauc) > 1 ? tcauc[2] .* topercents : tcauc[1] .* topercents, mem_1 = memauc[1] .* topercents, mem_max = length(memauc) > 1 ? memauc[2] .* topercents : memauc[1] .* topercents))
end

function cols2string!(df)
    for cname in names(df)
        map(x->df[x]=eltype(df[x]) == String ? df[x] : string.(df[x]) .* "%", names(df))
        if eltype(df[cname]) != String
            df[cname] = string.(df[cname]) .* "%"
        end
    end
end

mergeddf = vcat(mergeddf...)
PaperUtils.rounddf!(mergeddf, 1, 4:10)

printdf = mergeddf[mergeddf[:anom_ratio] .== 0.05, filter(x -> !(x in [:anom_ratio, :anom_seen]), names(mergeddf))]
cols2string!(printdf)

open("tables/table-full-results.tex", "w") do f
    write(f, PaperUtils.df2tex(printdf))
end
