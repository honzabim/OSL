push!(LOAD_PATH, pwd())
using KNNmem

function twoSpirals(N::Integer = 2000, degrees = 570, start = 90, noise = 0.2)
    # Generate "two spirals" dataset with N instances.
    # degrees controls the length of the spirals
    # start determines how far from the origin the spirals start, in degrees
    # noise displaces the instances from the spiral.
    #  0 is no noise, at 1 the spirals will start overlapping

    deg2rad = (2 * pi) / 360;
    start = start * deg2rad;

    N1 = floor(Int64, N / 2);
    N2 = N-N1;

    n = start + sqrt.(rand(N1)) * degrees * deg2rad;
    d1 = [-cos.(n) .* n + rand(N1) * noise sin.(n) .* n + rand(N1) * noise];

    n = start + sqrt.(rand(N1)) * degrees * deg2rad;
    d2 = [cos.(n) .* n + rand(N2) * noise -sin.(n) .* n + rand(N2) * noise];

    return [d1; d2], [zeros(Int64, N1); ones(Int64, N2)]
end

function halfPie(N::Integer = 2000)
    N1 = floor(Int64, N / 2);
    N2 = N-N1;

    d1 = [rand(N1) abs.(rand(N1))];
    d2 = [rand(N2) -abs.(rand(N2))];

    return [d1; d2], [zeros(Int64, N1); ones(Int64, N2)]
end

using Plots
pyplot() # Turns on the PyPlot backend

points, labels = twoSpirals();
#points, labels = halfPie();

# Plot initial data
# scatter(points[:, 1], points[:, 2], group=labels);

order = randperm(2000);
points = points[order, :];
labels = labels[order];

norms = zeros(Float64, 2000);
for i = 1:2000
    norms[i] = norm(points[i,:]);
end

points = [points norms]

#scatter(points[:, 1], points[:, 2], group=labels)

mem = KNNmemory(5, 3, 5, 2);

for i = 1:1500
    trainQuery!(mem, points[i, :] / norm(points[i, :]), labels[i]);
end

predictions = zeros(Int64, 500) - 1;

for i = 1501:2000
    predictions[i - 1500] = query(mem, points[i, :] / norm(points[i, :]));
end

scatter(points[1501:2000, 1], points[1501:2000, 2], group=(labels[1501:2000] .== predictions))
