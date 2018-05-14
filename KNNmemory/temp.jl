data = mapdata(sentence2ngrams,data)

function createmodel(data,target,k,l,lt)
m,k = Mill.reflectinmodel(data[1:2],idim -> FluxExtensions.layerbuilder(idim,k,k,l,"relu","relu",lt))
m = Mill.addlayer(m,Dense(k,size(target,1)))
m = Adapt.adapt(Float32,m)
m
end

# create function to evaluate single configuration
cr = p -> FluxExtensions.crossvalidate(data,target,
() -> createmodel(data,target,p...),
(x,y) -> FluxExtensions.logitcrossentropy(x,y),
100,20000)
# end
#find the best configuration over the grid
# (p,valerr,valerrs) = FluxExtensions.gridsearch(cr,product([5,10,20],[1,2,3],["Dense","ResDense"]))
df = FluxExtensions.gridsearch(cr,product([5,10,20],[1,2,3],["Dense","ResDense"]),["k","layers","layertype"],"recipes.csv")
df = by(df,[:k,:layers,:layertype], dff-> DataFrame(err = mean(dff[:err])))
i = indmin(df[:err])
p = (df[i,:k],df[i,:layers],String(df[i,:layertype]))
