function spirals(n)
  k = 3
  x = zeros(2,n*k)
  y = zeros(Int,n*k);
  r = linspace(0.0, 2.5, n)
  for i in 1:k
    t = linspace((i-1)*4, 4*(i),n) + 0.2randn(n)
    ix = (i-1)*n +1 : i*n
    x[:,ix] = vcat(transpose(r .* sin.(t)), transpose(r .* cos.(t)))
    y[ix] = i
  end
  (x,y)
end
