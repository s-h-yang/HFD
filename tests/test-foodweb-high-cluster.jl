using DelimitedFiles
using Printf
using Statistics

include("../utils.jl")
include("../fwHFD2.jl")

# Read data
labels = vec(readdlm(
    "../datasets/foodweb/foodweb_labels.txt", '\t', Int64, '\n'))
edges = Vector{Vector{Int64}}()
for line in eachline("../datasets/foodweb/foodweb_edges.txt")
    push!(edges, parse.(Int64, split(line,"\t")))
end
ne = length(edges)
nv = maximum(collect(Iterators.flatten(edges)))
incidence = [Int64[] for v in 1:nv]
for i in 1:ne
    for v in edges[i]
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H = HyperGraph(incidence, edges, degree, order, nv, ne)

# Run experiment for U-HFD
println("Running S-HFD on Florida Bay food network, High-level consumers")
println("===============================================================")
target = findall(x->x==3, labels) # Cluster High-level consumers is labelled 3
@printf("Size of cluster: %d\n", length(target))
@printf("Number of trials to run: %d\n", length(target))
println("===============================================================")
vol_T = sum(H.degree[target])
num_trials = length(target)
PR = zeros(num_trials)
RE = zeros(num_trials)
F1 = zeros(num_trials)
COND = zeros(num_trials)
for i in 1:num_trials
    Delta = zeros(Float64, H.nv)
    Delta[target[i]] = 5*vol_T
    cluster, COND[i], _, _ = fwHFD(H, Delta, sigma=0.0001, max_iters=30)
    PR[i], RE[i], F1[i] = compute_f1(cluster, target)
    @printf("Trial %d / %d, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n",
        i, num_trials, PR[i], RE[i], F1[i], COND[i])
end

# Write results to file
open("foodweb_high_SHFD.txt", "w") do f
    for i in 1:num_trials
        @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
    end
end
println("Job completed.")
@printf("Meidan F1 is %.2f, median conductance is %.2f\n",
    median(F1), median(COND))
