using Random
using Statistics
using DelimitedFiles
using Printf

include("../utils.jl")
include("../ucHFD.jl")
include("../LHQD/common.jl")
include("../LHQD/local-hyper.jl")
include("../LHQD/PageRank.jl")

# Read data
incidence = Vector{Vector{Int64}}()
for line in eachline("../datasets/mag-dual/mag-dual-incidence.txt")
    push!(incidence, parse.(Int64, split(line)))
end
edges = Vector{Vector{Int64}}()
for line in eachline("../datasets/mag-dual/mag-dual-edges.txt")
    push!(edges, parse.(Int64, split(line)))
end
labels = readdlm("../datasets/mag-dual/mag-dual-labels.txt", '\t', Int64, '\n')
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
nv = length(incidence)
ne = length(edges)
H = HyperGraph(incidence, edges, degree, order, nv, ne) # graph data structure for HFD
M = hyper2adjmat(H)
G = LH.graph(M, 1.0) # graph data structure for LH
Ga, deg = hypergraph_to_bipartite(G) # graph data structure for ACL

# Parameter setting for LH
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-10
max_iters = 1000000
L2 = LH.loss_type(2.0, 0.0)
L1 = LH.loss_type(1.4, 0.0)

# Parameter setting for HFD
sigma = 0.0001

# Run experiments
target_labels = ["data", "ml", "theory", "vision"]
r = init_r(H)
s = copy(r)
Delta = zeros(Float64, H.nv)

for label in target_labels

    target = vec(readdlm("../datasets/mag-dual/mag_"*label*"_papers.txt", '\t', Int64, '\n'))
    seeds = vec(readdlm("../datasets/seeds/mag_"*label*"_seeds.txt", '\t', Int64, '\n'))
    vol_T = sum(H.degree[target])
    num_trials = length(seeds)
    ratio = 1/length(target)
    kappa = 0.025*ratio # parameter for LH and ACL

    # HFD-2.0
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        fill!(Delta, 0.0)
        Delta[seeds[i]] = 3*vol_T
        cluster, COND[i], _, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=100, r=r, s=s)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %s, trial %d, method U-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("mag_"*label*"_UHFD.txt", "w") do f
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
        x,_,_ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L2,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %s, trial %d, method U-LH-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("mag_"*label*"_ULH2.0.txt", "w") do f
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
        x,_,_ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L1,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %s, trial %d, method U-LH-1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("mag_"*label*"_ULH1.4.txt", "w") do f
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
        x = PageRank.acl_diffusion(Ga,deg,[seeds[i]],gamma,kappa)
        x ./= deg
        x = x[1:size(G.H,2)]
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %s, trial %d, method ACL, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("mag_"*label*"_ACL.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

end
