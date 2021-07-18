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
labels = vec(readdlm("../datasets/contact-high-school/node-labels-contact-high-school.txt", '\t', Int64, '\n'))
edges = Vector{Vector{Int64}}()
for line in eachline("../datasets/contact-high-school/hyperedges-contact-high-school.txt")
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
H = HyperGraph(incidence, edges, degree, order, nv, ne)
M = hyper2adjmat(H)

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
for label in 1:9

    target = findall(x->x==label, labels)
    vol_T = sum(H.degree[target])
    seeds = copy(target)
    num_trials = length(seeds)
    ratio = 1/length(target)
    kappa = 0.25*ratio # parameter for LH and ACL

    # U-HFD
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = 3*vol_T
        cluster, COND[i], _, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=50)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %d, trial %d, method U-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_UHFD.txt", "w") do f
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
        Delta[seeds[i]] = 3*vol_T
        cluster, COND[i], _, _ = cardHFD(H, Delta, sigma=sigma, max_iters=20)
        PR[i], RE[i], F1[i] = compute_f1(cluster, target)
        @printf("Cluster %d, trial %d, method C-HFD, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_CHFD.txt", "w") do f
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
        x,_,_ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L2,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method U-LH-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_ULH2.0.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

    # U-LH-1.4
    G = LH.graph(M, 1.0)
    PR = zeros(num_trials)
    RE = zeros(num_trials)
    F1 = zeros(num_trials)
    COND = zeros(num_trials)
    for i in 1:num_trials
        x,_,_ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L1,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method U-LH-1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_ULH1.4.txt", "w") do f
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
        x,_,_ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L2,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method C-LH-2.0, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_CLH2.0.txt", "w") do f
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
        x,_,_ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L1,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method C-LH-1.4, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_CLH1.4.txt", "w") do f
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
        x = PageRank.acl_diffusion(Ga,deg,[seeds[i]],gamma,kappa)
        x ./= deg
        x = x[1:size(G.H,2)]
        COND[i], cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        PR[i], RE[i], F1[i] = PRF(target,cluster)
        @printf("Cluster %d, trial %d, method ACL, Pr %.4f, Re %.4f, F1 %.4f, Cond %.4f\n", label, i, PR[i], RE[i], F1[i], COND[i])
    end
    open("school_c"*string(label)*"_ACL.txt", "w") do f
        for i in 1:num_trials
            @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", PR[i], RE[i], F1[i], COND[i])
        end
    end

end
