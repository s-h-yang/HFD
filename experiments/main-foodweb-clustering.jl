using Random
using Statistics
using DelimitedFiles
using Printf

include("../utils.jl")
include("../ucHFD.jl")
include("../cardHFD2.jl")
include("../fwHFD2.jl")

# Read data
labels = vec(readdlm("../datasets/foodweb/foodweb_labels.txt", '\t', Int64, '\n'))
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

# Parameter setting for HFD
sigma = 0.0001
deltas = [20,10,5]

# Run experiments for unit, cardinality, and submodular cut-costs
target_labels = [1,2,3]

for (c,label) in enumerate(target_labels)

    target = findall(x->x==label, labels)
    vol_T = sum(H.degree[target])
    seeds = copy(target)
    num_trials = length(seeds)

    # U-HFD
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = deltas[c]*vol_T
        cluster, COND[i], _, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %d, trial %d, method U-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_UHFD.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # C-HFD
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = deltas[c]*vol_T
        cluster, COND[i], _, _ = cardHFD(H, Delta, sigma=sigma, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %d, trial %d, method C-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("food_c"*string(label)*"_CHFD.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # S-HFD
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = deltas[c]*vol_T
        cluster, COND[i], _, _ = fwHFD(H, Delta, sigma=sigma, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %d, trial %d, method S-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("food_c"*string(label)*"_SHFD.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

end
