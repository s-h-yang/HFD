using MAT
using Random
using Statistics
using DelimitedFiles
using Printf

include("../utils.jl")
include("../ucHFD.jl")
include("../fwHFD.jl")
include("LHQD/common.jl")
include("LHQD/local-hyper.jl")
include("LHQD/PageRank.jl")

# Read data
labels = vec(readdlm("datasets/foodweb/foodweb_labels.txt", '\t', Int64, '\n'))
edges = Vector{Vector{Int64}}()
for line in eachline("datasets/foodweb/foodweb_edges.txt")
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
H = HyperGraph(incidence, edges, degree, order, nv, ne) # graph data structure for HFD
M = hyper2adjmat(H)
G = LH.graph(M, 1.0) # data structure for LH
Ga, deg = hypergraph_to_bipartite(G) # data structure for ACL

target_labels = [1,2,3]
num_trials = 30
ratio = 0.01

# Parameter setting for LH
kappa = 0.025*ratio
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-6
max_iters2 = 1000000
max_iters1 = 10000 # reduce the number of iterations because LH-1.4 is too slow
L2 = LH.loss_type(2.0, 0.0)
L1 = LH.loss_type(1.4, 0.0)

# Parameter setting for HFD
sigma = 0.0001
deltas = [20,10,5]

# Run experiments for unit cut-cost
for (c,label) in enumerate(target_labels)
    target_cluster = findall(x->x==label, labels)
    vol_T = sum(H.degree[target_cluster])
    size_T = length(target_cluster)
    num_seeds = max(round(Int64,ratio*size_T), 5)
    kappa = 0.025*num_seeds/size_T # reset kappa this way improves LH for this dataset

    # HFD-2.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=30, p=2)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method HFD2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_HFD2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # HFD-4.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=30, p=4)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method HFD4.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_HFD4.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # LH-2.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L2,max_iters=max_iters2,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method LH2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_LH2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # LH-1.4
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L1,max_iters=max_iters1,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method LH1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_LH1.4.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # ACL
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        x = PageRank.acl_diffusion(Ga,deg,seeds,gamma,kappa)
        x ./= deg
        x = x[1:size(G.H,2)]
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method ACL, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_ACL.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end
end

# Run experiments for cardinality-based cut-cost
w = Dict{Set{Int64},Float64}()
w[Set{Int64}()] = 0.0
w[Set(1:4)] = 0.0
for i in 1:4
    w[Set([i])] = 0.5
    w[setdiff(Set(1:4),Set([i]))] = 0.5
end
w[Set([1,2])] = 1.0
w[Set([3,4])] = 1.0
w[Set([1,3])] = 1.0
w[Set([2,4])] = 1.0
w[Set([1,4])] = 1.0
w[Set([2,3])] = 1.0
R = [[0.5,0.5,-0.5,-0.5],
     [0.5,-0.5,0.5,-0.5],
     [0.5,-0.5,-0.5,0.5],
     [-0.5,0.5,0.5,-0.5],
     [-0.5,0.5,-0.5,0.5],
     [-0.5,-0.5,0.5,0.5],
     [0.5,-0.5,0.0,0.0],
     [0.5,0.0,-0.5,0.0],
     [0.5,0.0,0.0,-0.5],
     [0.0,0.5,-0.5,0.0],
     [0.0,0.5,0.0,-0.5],
     [0.0,0.0,0.5,-0.5],
     [-0.5,0.5,0.0,0.0],
     [-0.5,0.0,0.5,0.0],
     [-0.5,0.0,0.0,0.5],
     [0.0,-0.5,0.5,0.0],
     [0.0,-0.5,0.0,0.5],
     [0.0,0.0,-0.5,0.5]]

for (c,label) in enumerate(target_labels)
    target_cluster = findall(x->x==label, labels)
    vol_T = sum(H.degree[target_cluster])
    size_T = length(target_cluster)
    num_seeds = max(round(Int64,ratio*size_T), 5)

    # C-HFD-2.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _ = fwHFD(H, Delta, w, R, p=2, sigma=0.0001, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method C-HFD-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_CHFD2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # C-HFD-4.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _ = fwHFD(H, Delta, w, R, p=4, sigma=0.0001, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method C-HFD-4.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_CHFD4.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end
end

# Run experiments for submodular cut-cost
w = Dict{Set{Int64},Float64}()
w[Set{Int64}()] = 0.0
w[Set(1:4)] = 0.0
for i in 1:4
    w[Set([i])] = 0.5
    w[setdiff(Set(1:4),Set([i]))] = 0.5
end
w[Set([1,2])] = 0.0
w[Set([3,4])] = 0.0
w[Set([1,3])] = 1.0
w[Set([2,4])] = 1.0
w[Set([1,4])] = 1.0
w[Set([2,3])] = 1.0
R = [[0.5,-0.5,0.5,-0.5],
     [0.5,-0.5,-0.5,0.5],
     [-0.5,0.5,0.5,-0.5],
     [-0.5,0.5,-0.5,0.5],
     [0.5,-0.5,0.0,0.0],
     [-0.5,0.5,0.0,0.0],
     [0.0,0.0,0.5,-0.5],
     [0.0,0.0,-0.5,0.5]]

for (c,label) in enumerate(target_labels)
    target_cluster = findall(x->x==label, labels)
    vol_T = sum(H.degree[target_cluster])
    size_T = length(target_cluster)
    num_seeds = max(round(Int64,ratio*size_T), 5)

    # S-HFD-2.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _ = fwHFD(H, Delta, w, R, p=2, sigma=0.0001, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method S-HFD-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_SHFD2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # S-HFD-4.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _ = fwHFD(H, Delta, w, R, p=4, sigma=0.0001, max_iters=30)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method S-HFD-4.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("foodweb_c"*string(label)*"_multiseed_SHFD4.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end
end
