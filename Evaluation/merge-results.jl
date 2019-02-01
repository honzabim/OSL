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
    tcauc = NaN
    if size(twocapssel, 1) > 0
        tcauc = twocapssel[:auc]
    end

    pzpxsel = pzpxdf[(pzpxdf[:dataset] .== d) .& (pzpxdf[:anom_ratio] .== ar), :]
    pzauc = NaN
    pxauc = NaN
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
    if (length(pxvita) > 1)
        ds = [d, d]
        anr = [ar, ar]
        as = ["1", "max"]
    end
    # println("$ds, $anr, $as, $knn, $pxvita, $pzauc, $tcauc, $memauc")
    topercents = 100
    push!(mergeddf, DataFrame(dataset = ds, anom_ratio = anr, anom_seen = as, knn = knn .* topercents, svae_px = pxvita .* topercents, svae_pz = pzauc .* topercents, two_caps = tcauc .* topercents, mem = memauc .* topercents))
end

mergeddf = vcat(mergeddf...)
PaperUtils.rounddf!(mergeddf)
