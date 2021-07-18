using Random
using Statistics
using DelimitedFiles
using Printf

include("../utils.jl")
include("../ucHFD.jl")
include("../cardHFD2.jl")
include("../LHQD/common.jl")
include("../LHQD/local-hyper.jl")
include("../LHQD/PageRank.jl")

# Read data
labels = vec(readdlm("../datasets/trivago-clicks/node-labels-trivago-clicks.txt", '\t', Int64, '\n'))
edges = Vector{Vector{Int64}}()
for line in eachline("../datasets/trivago-clicks/hyperedges-trivago-clicks.txt")
    push!(edges, parse.(Int64, split(line,",")))
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

# Filter ground-truth clusters
target_labels = []
for i = 1:160
    target = findall(x->x==i, labels)
    if 100 <= length(target) <= 1000
        cond = uc_cond(H,target)
        if cond < 0.25
            push!(target_labels,i)
        end
    end
end

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

# Run experiments
for label in target_labels

    target_cluster = findall(x->x==label, labels)
    vol_T = sum(H.degree[target_cluster])
    size_T = length(target_cluster)
    num_seeds = max(round(Int64,ratio*size_T), 5)

    # U-HFD
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= 3*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=50)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target_cluster)
        @printf("Cluster %d, trial %d, method U-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("trivago_c"*string(label)*"_multiseed_UHFD.txt", "w") do f
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
        Random.seed!(i)
        seeds = target[randperm(size_T)[1:num_seeds]]
        vol_S = sum(H.degree[seeds])
        Delta = zeros(Float64, H.nv)
        Delta[seeds] .= 3*vol_T*H.degree[seeds]/vol_S
        cluster, COND[i], _, _ = cardHFD(H, Delta, sigma=sigma, max_iters=50)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %d, trial %d, method C-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("trivago_c"*string(label)*"_multiseed_CHFD.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # U-LH-2.0
    G = LH.graph(M, 1.0)
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L2,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method U-LH-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("trivago_c"*string(label)*"_multiseed_ULH2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # LH-1.4
    G = LH.graph(M, 1.0)
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target_cluster[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L1,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target_cluster,cluster)
        @printf("Cluster %d, trial %d, method U-LH-1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("trivago_c"*string(label)*"_multiseed_ULH1.4.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # C-LH-2.0
    G = LH.graph(M, convert(Float64, maximum(H.order)))
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L2,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method C-LH-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("trivago_c"*string(label)*"_multiseed_CLH2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # C-LH-1.4
    G = LH.graph(M, convert(Float64, maximum(H.order)))
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Random.seed!(i)
        seeds = target[randperm(size_T)[1:num_seeds]]
        x,_,_ = LH.lh_diffusion(G,seeds,gamma,kappa,rho,L1,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=num_seeds)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method C-LH-1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("trivago_c"*string(label)*"_multiseed_CLH1.4.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end


    # ACL
    G = LH.graph(M, 1.0)
    Ga, deg = hypergraph_to_bipartite(G)
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
    open("trivago_c"*string(label)*"_multiseed_ACL.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

end
