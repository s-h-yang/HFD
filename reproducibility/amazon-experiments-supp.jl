using MAT
using Random
using Statistics
using DelimitedFiles
using Printf

include("../utils.jl");
include("../ucHFD.jl")
include("LHQD/common.jl")
include("LHQD/local-hyper.jl")
include("LHQD/PageRank.jl")

# Read data
M = matread("datasets/AmazonReview5core_H.mat")
incidence, edges = mat2list(M["H"])
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
nv = length(degree)
ne = length(order)
H = HyperGraph(incidence, edges, degree, order, nv, ne) # graph data structure for LH
G = LH.graph(M["H"], 1.0) # graph data structure for HFD
Ga, deg = hypergraph_to_bipartite(G) # graph data structure for ACL

target_labels = [1,2,3,12,18,15,17,24,25]
num_trials = 30
ratio = 0.01

# Parameter setting for LH
kappa = 0.025*ratio
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-10
max_iters = 1000000
L2 = LH.loss_type(2.0, 0.0)
L1 = LH.loss_type(1.4, 0.0)

# Parameter setting for HFD
sigma = 0.0001
deltas = [200,200,200,200,200,50,50,50,50]

# Run experiments
r = init_r(H)
s = copy(r)
Delta = zeros(Float64, H.nv)

for (c,label) in enumerate(target_labels)
    target_cluster = findall(x->x==label, M["NodeLabels"])
    vol_T = sum(H.degree[target_cluster])
    size_T = length(target_cluster)
    num_seeds = max(round(Int64,ratio*size_T), 5)

    # HFD-2.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        fill!(Delta, 0.0)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=30, p=2, r=r, s=s)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method HFD2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("amazon_c"*string(label)*"_multiseed_HFD2.0.txt", "w") do f
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
        seeds = target[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        fill!(Delta, 0.0)
        Delta[seeds] .= deltas[c]*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=30, p=4, r=r, s=s)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method HFD4.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("amazon_c"*string(label)*"_multiseed_HFD4.0.txt", "w") do f
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
        seeds = target[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L2,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method LH2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("amazon_c"*string(label)*"_multiseed_LH2.0.txt", "w") do f
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
        seeds = target[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L1,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method LH1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("amazon_c"*string(label)*"_multiseed_LH1.4.txt", "w") do f
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
        seeds = target[randperm(size_T)[1:num_seeds]]
        x = PageRank.acl_diffusion(Ga,deg,seeds,gamma,kappa)
        x ./= deg
        x = x[1:size(G.H,2)]
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method ACL, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("amazon_c"*string(label)*"_multiseed_ACL.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end
end
