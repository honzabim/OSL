module KNNmem

using Flux

# TODO export neco, necoJinyho

gridSearch(f, parameters) =
    errs = map(p -> (p, f(p)), parameters)
    iMin = indmin(errs)
    return(errs[iMin][1], errs[iMin][2], errs)
end

function gridsearch(f, reduceFunc, parameters)
    errs = map(p -> (p, f(p)), parameters)
    iMin = indmin(map(v -> reduceFunc(v[2]), errs))
    return(errs[iMin][1], errs[iMin][2], errs)
end
