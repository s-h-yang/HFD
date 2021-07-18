using Random
using Statistics
using DelimitedFiles
using Printf

using PyCall, PyPlot
plt = pyimport("matplotlib.pyplot")

include("../utils.jl");
include("../ucHFD.jl")
include("../cardHFD2.jl")
include("../LHQD/common.jl")
include("../LHQD/local-hyper.jl")
include("../LHQD/PageRank.jl")

true_ccond = Vector{Float64}()

edges = readdlm("../datasets/synthetic/hyperedges_k3HSBM_cond30.txt", '\t', Int, '\n')
edges = [c[:] for c in eachrow(edges)]
ne = length(edges)
nv = 100
incidence = [Int64[] for v in 1:nv]
for (i,e) in enumerate(edges)
    for v in e
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H = HyperGraph(incidence, edges, degree, order, nv, ne)
M = hyper2adjmat(H)
target = collect(1:50)
vol_T = sum(H.degree[target])
seeds = copy(target)
num_trials = length(seeds)
sigma = 0.01
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-16
max_iters = 1000000
ratio = 1/length(target)
deltas = 2.0.^collect(1:12)
L = LH.loss_type(2.0, 0.0)

append!(true_ccond, card_cond(H, collect(1:50)))

k3_UHFD_pr = zeros(num_trials)
k3_UHFD_re = zeros(num_trials)
k3_UHFD_f1 = zeros(num_trials)
k3_UHFD_ucond = zeros(num_trials)
k3_UHFD_ccond = zeros(num_trials)
k3_CHFD_pr = zeros(num_trials)
k3_CHFD_re = zeros(num_trials)
k3_CHFD_f1 = zeros(num_trials)
k3_CHFD_ucond = zeros(num_trials)
k3_CHFD_ccond = zeros(num_trials)
k3_ULH_pr = zeros(num_trials)
k3_ULH_re = zeros(num_trials)
k3_ULH_f1 = zeros(num_trials)
k3_ULH_ucond = zeros(num_trials)
k3_ULH_ccond = zeros(num_trials)
k3_CLH_pr = zeros(num_trials)
k3_CLH_re = zeros(num_trials)
k3_CLH_f1 = zeros(num_trials)
k3_CLH_ucond = zeros(num_trials)
k3_CLH_ccond = zeros(num_trials)
k3_ACL_pr = zeros(num_trials)
k3_ACL_re = zeros(num_trials)
k3_ACL_f1 = zeros(num_trials)
k3_ACL_ucond = zeros(num_trials)
k3_ACL_ccond = zeros(num_trials)

for i in 1:num_trials

    @printf("k = 3, trial %d\n", i)

    # U-HFD and C-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=20)
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k3_UHFD_pr[i], k3_UHFD_re[i], k3_UHFD_f1[i] = compute_f1(cluster, target)
    k3_UHFD_ucond[i] = uc_cond(H, cluster)
    k3_UHFD_ccond[i] = card_cond(H, cluster)

    # U-LH
    G = LH.graph(M, 1.0)
    kappa = 0.35*ratio
    x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
    k3_ULH_pr[i], k3_ULH_re[i], k3_ULH_f1[i] = compute_f1(cluster, target)
    k3_ULH_ucond[i] = uc_cond(H, cluster)
    k3_ULH_ccond[i] = card_cond(H, cluster)

    # C-LH
    best_cond = 1.0
    best_cluster = Vector{Int64}()
    for delta in deltas
        G = LH.graph(M, delta)
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        temp_cond = card_cond(H, cluster)
        if temp_cond < best_cond
            best_cond = temp_cond
            best_cluster = copy(cluster)
        end
    end
    k3_CLH_pr[i], k3_CLH_re[i], k3_CLH_f1[i] = compute_f1(best_cluster, target)
    k3_CLH_ucond[i] = uc_cond(H, best_cluster)
    k3_CLH_ccond[i] = card_cond(H, best_cluster)

    # ACL
    G = LH.graph(M, 1.0)
    Ga, deg = hypergraph_to_bipartite(G)
    kappa = 0.025*ratio
    x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
    x = x[1:size(G.H,2)]
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k3_ACL_pr[i], k3_ACL_re[i], k3_ACL_f1[i] = compute_f1(cluster, target)
    k3_ACL_ucond[i] = uc_cond(H, cluster)
    k3_ACL_ccond[i] = card_cond(H, cluster)
end


edges = readdlm("../datasets/synthetic/hyperedges_k4HSBM_cond30.txt", '\t', Int, '\n')
edges = [c[:] for c in eachrow(edges)]
ne = length(edges)
nv = 100
incidence = [Int64[] for v in 1:nv]
for (i,e) in enumerate(edges)
    for v in e
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H = HyperGraph(incidence, edges, degree, order, nv, ne);
M = hyper2adjmat(H)
target = collect(1:50)
vol_T = sum(H.degree[target])
seeds = copy(target)
num_trials = length(seeds)
sigma = 0.01
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-16
max_iters = 1000000
ratio = 1/length(target)
deltas = 2.0.^collect(1:12)
L = LH.loss_type(2.0, 0.0)

append!(true_ccond, card_cond(H, collect(1:50)))

k4_UHFD_pr = zeros(num_trials)
k4_UHFD_re = zeros(num_trials)
k4_UHFD_f1 = zeros(num_trials)
k4_UHFD_ucond = zeros(num_trials)
k4_UHFD_ccond = zeros(num_trials)
k4_CHFD_pr = zeros(num_trials)
k4_CHFD_re = zeros(num_trials)
k4_CHFD_f1 = zeros(num_trials)
k4_CHFD_ucond = zeros(num_trials)
k4_CHFD_ccond = zeros(num_trials)
k4_ULH_pr = zeros(num_trials)
k4_ULH_re = zeros(num_trials)
k4_ULH_f1 = zeros(num_trials)
k4_ULH_ucond = zeros(num_trials)
k4_ULH_ccond = zeros(num_trials)
k4_CLH_pr = zeros(num_trials)
k4_CLH_re = zeros(num_trials)
k4_CLH_f1 = zeros(num_trials)
k4_CLH_ucond = zeros(num_trials)
k4_CLH_ccond = zeros(num_trials)
k4_ACL_pr = zeros(num_trials)
k4_ACL_re = zeros(num_trials)
k4_ACL_f1 = zeros(num_trials)
k4_ACL_ucond = zeros(num_trials)
k4_ACL_ccond = zeros(num_trials)

for i in 1:num_trials

    @printf("k = 4, trial %d\n", i)

    # U-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=10)
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k4_UHFD_pr[i], k4_UHFD_re[i], k4_UHFD_f1[i] = compute_f1(cluster, target)
    k4_UHFD_ucond[i] = uc_cond(H, cluster)
    k4_UHFD_ccond[i] = card_cond(H, cluster)

    # C-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _ = cardHFD(H, Delta, sigma=sigma, max_iters=10)
    cluster, _ = card_sweepcut(H, x./H.degree)
    k4_CHFD_pr[i], k4_CHFD_re[i], k4_CHFD_f1[i] = compute_f1(cluster, target)
    k4_CHFD_ucond[i] = uc_cond(H, cluster)
    k4_CHFD_ccond[i] = card_cond(H, cluster)

    # U-LH
    G = LH.graph(M, 1.0)
    kappa = 0.35*ratio
    x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
    k4_ULH_pr[i], k4_ULH_re[i], k4_ULH_f1[i] = compute_f1(cluster, target)
    k4_ULH_ucond[i] = uc_cond(H, cluster)
    k4_ULH_ccond[i] = card_cond(H, cluster)

    # C-LH
    best_cond = 1.0
    best_cluster = Vector{Int64}()
    for delta in deltas
        G = LH.graph(M, delta)
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        temp_cond = card_cond(H, cluster)
        if temp_cond < best_cond
            best_cond = temp_cond
            best_cluster = copy(cluster)
        end
    end
    k4_CLH_pr[i], k4_CLH_re[i], k4_CLH_f1[i] = compute_f1(best_cluster, target)
    k4_CLH_ucond[i] = uc_cond(H, best_cluster)
    k4_CLH_ccond[i] = card_cond(H, best_cluster)

    # ACL
    G = LH.graph(M, 1.0)
    Ga, deg = hypergraph_to_bipartite(G)
    kappa = 0.025*ratio
    x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
    x = x[1:size(G.H,2)]
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k4_ACL_pr[i], k4_ACL_re[i], k4_ACL_f1[i] = compute_f1(cluster, target)
    k4_ACL_ucond[i] = uc_cond(H, cluster)
    k4_ACL_ccond[i] = card_cond(H, cluster)
end


edges = readdlm("../datasets/synthetic/hyperedges_k5HSBM_cond30.txt", '\t', Int, '\n')
edges = [c[:] for c in eachrow(edges)]
ne = length(edges)
nv = 100
incidence = [Int64[] for v in 1:nv]
for (i,e) in enumerate(edges)
    for v in e
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H = HyperGraph(incidence, edges, degree, order, nv, ne)
M = hyper2adjmat(H)
target = collect(1:50)
vol_T = sum(H.degree[target])
seeds = copy(target)
num_trials = length(seeds)
sigma = 0.01
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-16
max_iters = 1000000
ratio = 1/length(target)
deltas = 2.0.^collect(1:12)
L = LH.loss_type(2.0, 0.0)

append!(true_ccond, card_cond(H, collect(1:50)))

k5_UHFD_pr = zeros(num_trials)
k5_UHFD_re = zeros(num_trials)
k5_UHFD_f1 = zeros(num_trials)
k5_UHFD_ucond = zeros(num_trials)
k5_UHFD_ccond = zeros(num_trials)
k5_CHFD_pr = zeros(num_trials)
k5_CHFD_re = zeros(num_trials)
k5_CHFD_f1 = zeros(num_trials)
k5_CHFD_ucond = zeros(num_trials)
k5_CHFD_ccond = zeros(num_trials)
k5_ULH_pr = zeros(num_trials)
k5_ULH_re = zeros(num_trials)
k5_ULH_f1 = zeros(num_trials)
k5_ULH_ucond = zeros(num_trials)
k5_ULH_ccond = zeros(num_trials)
k5_CLH_pr = zeros(num_trials)
k5_CLH_re = zeros(num_trials)
k5_CLH_f1 = zeros(num_trials)
k5_CLH_ucond = zeros(num_trials)
k5_CLH_ccond = zeros(num_trials)
k5_ACL_pr = zeros(num_trials)
k5_ACL_re = zeros(num_trials)
k5_ACL_f1 = zeros(num_trials)
k5_ACL_ucond = zeros(num_trials)
k5_ACL_ccond = zeros(num_trials)

for i in 1:num_trials

    @printf("k = 5, trial %d\n", i)

    # U-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=10)
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k5_UHFD_pr[i], k5_UHFD_re[i], k5_UHFD_f1[i] = compute_f1(cluster, target)
    k5_UHFD_ucond[i] = uc_cond(H, cluster)
    k5_UHFD_ccond[i] = card_cond(H, cluster)

    # C-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _ = cardHFD(H, Delta, sigma=sigma, max_iters=10)
    cluster, _ = card_sweepcut(H, x./H.degree)
    k5_CHFD_pr[i], k5_CHFD_re[i], k5_CHFD_f1[i] = compute_f1(cluster, target)
    k5_CHFD_ucond[i] = uc_cond(H, cluster)
    k5_CHFD_ccond[i] = card_cond(H, cluster)

    # U-LH
    G = LH.graph(M, 1.0)
    kappa = 0.35*ratio
    x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
    k5_ULH_pr[i], k5_ULH_re[i], k5_ULH_f1[i] = compute_f1(cluster, target)
    k5_ULH_ucond[i] = uc_cond(H, cluster)
    k5_ULH_ccond[i] = card_cond(H, cluster)

    # C-LH
    best_cond = 1.0
    best_cluster = Vector{Int64}()
    for delta in deltas
        G = LH.graph(M, delta)
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        temp_cond = card_cond(H, cluster)
        if temp_cond < best_cond
            best_cond = temp_cond
            best_cluster = copy(cluster)
        end
    end
    k5_CLH_pr[i], k5_CLH_re[i], k5_CLH_f1[i] = compute_f1(best_cluster, target)
    k5_CLH_ucond[i] = uc_cond(H, best_cluster)
    k5_CLH_ccond[i] = card_cond(H, best_cluster)

    # ACL
    G = LH.graph(M, 1.0)
    Ga, deg = hypergraph_to_bipartite(G)
    kappa = 0.025*ratio
    x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
    x = x[1:size(G.H,2)]
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k5_ACL_pr[i], k5_ACL_re[i], k5_ACL_f1[i] = compute_f1(cluster, target)
    k5_ACL_ucond[i] = uc_cond(H, cluster)
    k5_ACL_ccond[i] = card_cond(H, cluster)
end

edges = readdlm("../datasets/synthetic/hyperedges_k6HSBM_cond30.txt", '\t', Int, '\n')
edges = [c[:] for c in eachrow(edges)]
ne = length(edges)
nv = 100
incidence = [Int64[] for v in 1:nv]
for (i,e) in enumerate(edges)
    for v in e
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H = HyperGraph(incidence, edges, degree, order, nv, ne);
M = hyper2adjmat(H)
target = collect(1:50)
vol_T = sum(H.degree[target])
seeds = copy(target)
num_trials = length(seeds)
sigma = 0.005
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-16
max_iters = 1000000
ratio = 1/length(target)
deltas = 2.0.^collect(1:12)
L = LH.loss_type(2.0, 0.0)

append!(true_ccond, card_cond(H, collect(1:50)))

k6_UHFD_pr = zeros(num_trials)
k6_UHFD_re = zeros(num_trials)
k6_UHFD_f1 = zeros(num_trials)
k6_UHFD_ucond = zeros(num_trials)
k6_UHFD_ccond = zeros(num_trials)
k6_CHFD_pr = zeros(num_trials)
k6_CHFD_re = zeros(num_trials)
k6_CHFD_f1 = zeros(num_trials)
k6_CHFD_ucond = zeros(num_trials)
k6_CHFD_ccond = zeros(num_trials)
k6_ULH_pr = zeros(num_trials)
k6_ULH_re = zeros(num_trials)
k6_ULH_f1 = zeros(num_trials)
k6_ULH_ucond = zeros(num_trials)
k6_ULH_ccond = zeros(num_trials)
k6_CLH_pr = zeros(num_trials)
k6_CLH_re = zeros(num_trials)
k6_CLH_f1 = zeros(num_trials)
k6_CLH_ucond = zeros(num_trials)
k6_CLH_ccond = zeros(num_trials)
k6_ACL_pr = zeros(num_trials)
k6_ACL_re = zeros(num_trials)
k6_ACL_f1 = zeros(num_trials)
k6_ACL_ucond = zeros(num_trials)
k6_ACL_ccond = zeros(num_trials)

for i in 1:num_trials

    @printf("k = 6, trial %d\n", i)

    # U-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=10)
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k6_UHFD_pr[i], k6_UHFD_re[i], k6_UHFD_f1[i] = compute_f1(cluster, target)
    k6_UHFD_ucond[i] = uc_cond(H, cluster)
    k6_UHFD_ccond[i] = card_cond(H, cluster)

    # C-HFD
    Delta = zeros(Float64, H.nv)
    Delta[seeds[i]] = 3*vol_T
    _, _, x, _ = cardHFD(H, Delta, sigma=sigma, max_iters=10)
    cluster, _ = card_sweepcut(H, x./H.degree)
    k6_CHFD_pr[i], k6_CHFD_re[i], k6_CHFD_f1[i] = compute_f1(cluster, target)
    k6_CHFD_ucond[i] = uc_cond(H, cluster)
    k6_CHFD_ccond[i] = card_cond(H, cluster)

    # U-LH
    G = LH.graph(M, 1.0)
    kappa = 0.35*ratio
    x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
    k6_ULH_pr[i], k6_ULH_re[i], k6_ULH_f1[i] = compute_f1(cluster, target)
    k6_ULH_ucond[i] = uc_cond(H, cluster)
    k6_ULH_ccond[i] = card_cond(H, cluster)

    # C-LH
    best_cond = 1.0
    best_cluster = Vector{Int64}()
    for delta in deltas
        G = LH.graph(M, delta)
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        temp_cond = card_cond(H, cluster)
        if temp_cond < best_cond
            best_cond = temp_cond
            best_cluster = copy(cluster)
        end
    end
    k6_CLH_pr[i], k6_CLH_re[i], k6_CLH_f1[i] = compute_f1(best_cluster, target)
    k6_CLH_ucond[i] = uc_cond(H, best_cluster)
    k6_CLH_ccond[i] = card_cond(H, best_cluster)

    # ACL
    G = LH.graph(M, 1.0)
    Ga, deg = hypergraph_to_bipartite(G)
    kappa = 0.025*ratio
    x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
    x = x[1:size(G.H,2)]
    cluster, _ = uc_sweepcut(H, x./H.degree)
    k6_ACL_pr[i], k6_ACL_re[i], k6_ACL_f1[i] = compute_f1(cluster, target)
    k6_ACL_ucond[i] = uc_cond(H, cluster)
    k6_ACL_ccond[i] = card_cond(H, cluster)
end


UHFD_f1_mean = [mean(k3_UHFD_f1),mean(k4_UHFD_f1),mean(k5_UHFD_f1),mean(k6_UHFD_f1)]
CHFD_f1_mean = [mean(k3_UHFD_f1),mean(k4_CHFD_f1),mean(k5_CHFD_f1),mean(k6_CHFD_f1)]
ULH_f1_mean = [mean(k3_ULH_f1),mean(k4_ULH_f1),mean(k5_ULH_f1),mean(k6_ULH_f1)]
CLH_f1_mean = [mean(k3_CLH_f1),mean(k4_CLH_f1),mean(k5_CLH_f1),mean(k6_CLH_f1)]
ACL_f1_mean = [mean(k3_ACL_f1),mean(k4_ACL_f1),mean(k5_ACL_f1),mean(k6_ACL_f1)]
UHFD_f1_std = [std(k3_UHFD_f1),std(k4_UHFD_f1),std(k5_UHFD_f1),std(k6_UHFD_f1)]
CHFD_f1_std = [std(k3_UHFD_f1),std(k4_CHFD_f1),std(k5_CHFD_f1),std(k6_CHFD_f1)]
ULH_f1_std = [std(k3_ULH_f1),std(k4_ULH_f1),std(k5_ULH_f1),std(k6_ULH_f1)]
CLH_f1_std = [std(k3_CLH_f1),std(k4_CLH_f1),std(k5_CLH_f1),std(k6_CLH_f1)]
ACL_f1_std = [std(k3_ACL_f1),std(k4_ACL_f1),std(k5_ACL_f1),std(k6_ACL_f1)]
UHFD_f1_median = [median(k3_UHFD_f1),median(k4_UHFD_f1),median(k5_UHFD_f1),median(k6_UHFD_f1)]
CHFD_f1_median = [median(k3_UHFD_f1),median(k4_CHFD_f1),median(k5_CHFD_f1),median(k6_CHFD_f1)]
ULH_f1_median = [median(k3_ULH_f1),median(k4_ULH_f1),median(k5_ULH_f1),median(k6_ULH_f1)]
CLH_f1_median = [median(k3_CLH_f1),median(k4_CLH_f1),median(k5_CLH_f1),median(k6_CLH_f1)]
ACL_f1_median = [median(k3_ACL_f1),median(k4_ACL_f1),median(k5_ACL_f1),median(k6_ACL_f1)]
UHFD_f1_p25 = [quantile(k3_UHFD_f1,0.25),quantile(k4_UHFD_f1,0.25),quantile(k5_UHFD_f1,0.25),quantile(k6_UHFD_f1,0.25)]
CHFD_f1_p25 = [quantile(k3_UHFD_f1,0.25),quantile(k4_CHFD_f1,0.25),quantile(k5_CHFD_f1,0.25),quantile(k6_CHFD_f1,0.25)]
ULH_f1_p25 = [quantile(k3_ULH_f1,0.25),quantile(k4_ULH_f1,0.25),quantile(k5_ULH_f1,0.25),quantile(k6_ULH_f1,0.25)]
CLH_f1_p25 = [quantile(k3_CLH_f1,0.25),quantile(k4_CLH_f1,0.25),quantile(k5_CLH_f1,0.25),quantile(k6_CLH_f1,0.25)]
ACL_f1_p25 = [quantile(k3_ACL_f1,0.25),quantile(k4_ACL_f1,0.25),quantile(k5_ACL_f1,0.25),quantile(k6_ACL_f1,0.25)]
UHFD_f1_p75 = [quantile(k3_UHFD_f1,0.75),quantile(k4_UHFD_f1,0.75),quantile(k5_UHFD_f1,0.75),quantile(k6_UHFD_f1,0.75)]
CHFD_f1_p75 = [quantile(k3_UHFD_f1,0.75),quantile(k4_CHFD_f1,0.75),quantile(k5_CHFD_f1,0.75),quantile(k6_CHFD_f1,0.75)]
ULH_f1_p75 = [quantile(k3_ULH_f1,0.75),quantile(k4_ULH_f1,0.75),quantile(k5_ULH_f1,0.75),quantile(k6_ULH_f1,0.75)]
CLH_f1_p75 = [quantile(k3_CLH_f1,0.75),quantile(k4_CLH_f1,0.75),quantile(k5_CLH_f1,0.75),quantile(k6_CLH_f1,0.75)]
ACL_f1_p75 = [quantile(k3_ACL_f1,0.75),quantile(k4_ACL_f1,0.75),quantile(k5_ACL_f1,0.75),quantile(k6_ACL_f1,0.75)]


fig, ax = plt.subplots()
plt.grid(linestyle="-", linewidth = 0.5, axis = "y")
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex=true)
x_pos = [0,3,6,9]
ax.plot(x_pos.-0.5, UHFD_f1_median, linestyle="--", linewidth=2, color="tab:blue", alpha=0.3)
ax.plot(x_pos.-0.25, CHFD_f1_median, linestyle="--", linewidth=2, color="tab:red", alpha=0.3)
ax.plot(x_pos, ULH_f1_median, linestyle="--", linewidth=2, color="tab:purple", alpha=0.3)
ax.plot(x_pos.+0.25, CLH_f1_median, linestyle="--", linewidth=2, color="tab:orange", alpha=0.3)
ax.plot(x_pos.+0.5, ACL_f1_median, linestyle="--", linewidth=2, color="tab:green", alpha=0.3)
ax.errorbar(x_pos.-0.5, UHFD_f1_median, yerr=[UHFD_f1_median-UHFD_f1_p25,UHFD_f1_p75-UHFD_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="o", markersize=12, color="tab:blue", label="U-HFD")
ax.errorbar(x_pos.-0.25, CHFD_f1_median, yerr=[CHFD_f1_median-CHFD_f1_p25,CHFD_f1_p75-CHFD_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="^", markersize=12, color="tab:red", label="C-HFD")
ax.errorbar(x_pos, ULH_f1_median, yerr=[ULH_f1_median-ULH_f1_p25,ULH_f1_p75-ULH_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="*", markersize=15, color="tab:purple", label="U-LH")
ax.errorbar(x_pos.+0.25, CLH_f1_median, yerr=[CLH_f1_median-CLH_f1_p25,CLH_f1_p75-CLH_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="D", markersize=11, color="tab:orange", label="C-LH")
ax.errorbar(x_pos.+0.5, ACL_f1_median, yerr=[ACL_f1_median-ACL_f1_p25,ACL_f1_p75-ACL_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="X", markersize=13, color="tab:green", label="ACL")
ax.set_xticks(x_pos)
ax.set_yticks(range(0.8,1,length=6))
ax.set_xticklabels([L"$k=3$", L"$k=4$", L"$k=5$", L"$k=6$"], size=28)
ax.set_yticklabels(["0.80", "0.84", "0.86", "0.92", "0.96", "1.00"], size=18, fontname="Times New Roman")
ax.tick_params(axis="x", which="both",length=0)
plt.legend(fontsize=18, handletextpad=0.1)
plt.ylabel("F1 score", fontsize=30, fontname="Times New Roman")
plt.savefig("f1_edgesize_cond30.pdf", bbox_inches="tight", format="pdf", dpi=400)

k3_UHFD_ccond_relerr = k3_UHFD_ccond ./ true_ccond[1]
k3_ULH_ccond_relerr = k3_ULH_ccond ./ true_ccond[1]
k3_CLH_ccond_relerr = k3_CLH_ccond ./ true_ccond[1]
k3_ACL_ccond_relerr = k3_ACL_ccond ./ true_ccond[1]
k4_UHFD_ccond_relerr = k4_UHFD_ccond ./ true_ccond[2]
k4_CHFD_ccond_relerr = k4_CHFD_ccond ./ true_ccond[2]
k4_ULH_ccond_relerr = k4_ULH_ccond ./ true_ccond[2]
k4_CLH_ccond_relerr = k4_CLH_ccond ./ true_ccond[2]
k4_ACL_ccond_relerr = k4_ACL_ccond ./ true_ccond[2]
k5_UHFD_ccond_relerr = k5_UHFD_ccond ./ true_ccond[3]
k5_CHFD_ccond_relerr = k5_CHFD_ccond ./ true_ccond[3]
k5_ULH_ccond_relerr = k5_ULH_ccond ./ true_ccond[3]
k5_CLH_ccond_relerr = k5_CLH_ccond ./ true_ccond[3]
k5_ACL_ccond_relerr = k5_ACL_ccond ./ true_ccond[3]
k6_UHFD_ccond_relerr = k6_UHFD_ccond ./ true_ccond[4]
k6_CHFD_ccond_relerr = k6_CHFD_ccond ./ true_ccond[4]
k6_ULH_ccond_relerr = k6_ULH_ccond ./ true_ccond[4]
k6_CLH_ccond_relerr = k6_CLH_ccond ./ true_ccond[4]
k6_ACL_ccond_relerr = k6_ACL_ccond ./ true_ccond[4]
UHFD_ccond_relerr_median = [median(k3_UHFD_ccond_relerr),median(k4_UHFD_ccond_relerr),median(k5_UHFD_ccond_relerr),median(k6_UHFD_ccond_relerr)];
CHFD_ccond_relerr_median = [median(k3_UHFD_ccond_relerr),median(k4_CHFD_ccond_relerr),median(k5_CHFD_ccond_relerr),median(k6_CHFD_ccond_relerr)];
ULH_ccond_relerr_median = [median(k3_ULH_ccond_relerr),median(k4_ULH_ccond_relerr),median(k5_ULH_ccond_relerr),median(k6_ULH_ccond_relerr)];
CLH_ccond_relerr_median = [median(k3_CLH_ccond_relerr),median(k4_CLH_ccond_relerr),median(k5_CLH_ccond_relerr),median(k6_CLH_ccond_relerr)];
ACL_ccond_relerr_median = [median(k3_ACL_ccond_relerr),median(k4_ACL_ccond_relerr),median(k5_ACL_ccond_relerr),median(k6_ACL_ccond_relerr)];
UHFD_ccond_relerr_p25 = [quantile(k3_UHFD_ccond_relerr,0.25),quantile(k4_UHFD_ccond_relerr,0.25),quantile(k5_UHFD_ccond_relerr,0.25),quantile(k6_UHFD_ccond_relerr,0.25)];
CHFD_ccond_relerr_p25 = [quantile(k3_UHFD_ccond_relerr,0.25),quantile(k4_CHFD_ccond_relerr,0.25),quantile(k5_CHFD_ccond_relerr,0.25),quantile(k6_CHFD_ccond_relerr,0.25)];
ULH_ccond_relerr_p25 = [quantile(k3_ULH_ccond_relerr,0.25),quantile(k4_ULH_ccond_relerr,0.25),quantile(k5_ULH_ccond_relerr,0.25),quantile(k6_ULH_ccond_relerr,0.25)];
CLH_ccond_relerr_p25 = [quantile(k3_CLH_ccond_relerr,0.25),quantile(k4_CLH_ccond_relerr,0.25),quantile(k5_CLH_ccond_relerr,0.25),quantile(k6_CLH_ccond_relerr,0.25)];
ACL_ccond_relerr_p25 = [quantile(k3_ACL_ccond_relerr,0.25),quantile(k4_ACL_ccond_relerr,0.25),quantile(k5_ACL_ccond_relerr,0.25),quantile(k6_ACL_ccond_relerr,0.25)];
UHFD_ccond_relerr_p75 = [quantile(k3_UHFD_ccond_relerr,0.75),quantile(k4_UHFD_ccond_relerr,0.75),quantile(k5_UHFD_ccond_relerr,0.75),quantile(k6_UHFD_ccond_relerr,0.75)];
CHFD_ccond_relerr_p75 = [quantile(k3_UHFD_ccond_relerr,0.75),quantile(k4_CHFD_ccond_relerr,0.75),quantile(k5_CHFD_ccond_relerr,0.75),quantile(k6_CHFD_ccond_relerr,0.75)];
ULH_ccond_relerr_p75 = [quantile(k3_ULH_ccond_relerr,0.75),quantile(k4_ULH_ccond_relerr,0.75),quantile(k5_ULH_ccond_relerr,0.75),quantile(k6_ULH_ccond_relerr,0.75)];
CLH_ccond_relerr_p75 = [quantile(k3_CLH_ccond_relerr,0.75),quantile(k4_CLH_ccond_relerr,0.75),quantile(k5_CLH_ccond_relerr,0.75),quantile(k6_CLH_ccond_relerr,0.75)];
ACL_ccond_relerr_p75 = [quantile(k3_ACL_ccond_relerr,0.75),quantile(k4_ACL_ccond_relerr,0.75),quantile(k5_ACL_ccond_relerr,0.75),quantile(k6_ACL_ccond_relerr,0.75)];

fig, ax = plt.subplots()
plt.grid(linestyle="-", linewidth = 0.5, axis = "y")
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
x_pos = [0,3,6,9]
ax.plot(x_pos.-0.5, UHFD_ccond_relerr_median, linestyle="--", linewidth=2, color="tab:blue", alpha=0.3)
ax.plot(x_pos.-0.25, CHFD_ccond_relerr_median, linestyle="--", linewidth=2, color="tab:red", alpha=0.3)
ax.plot(x_pos, ULH_ccond_relerr_median, linestyle="--", linewidth=2, color="tab:purple", alpha=0.3)
ax.plot(x_pos.+0.25, CLH_ccond_relerr_median, linestyle="--", linewidth=2, color="tab:orange", alpha=0.3)
ax.plot(x_pos.+0.5, ACL_ccond_relerr_median, linestyle="--", linewidth=2, color="tab:green", alpha=0.3)
ax.errorbar(x_pos.-0.5, UHFD_ccond_relerr_median, yerr=[UHFD_ccond_relerr_median-UHFD_ccond_relerr_p25,UHFD_ccond_relerr_p75-UHFD_ccond_relerr_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="o", markersize=12, color="tab:blue", label="U-HFD")
ax.errorbar(x_pos.-0.25, CHFD_ccond_relerr_median, yerr=[CHFD_ccond_relerr_median-CHFD_ccond_relerr_p25,CHFD_ccond_relerr_p75-CHFD_ccond_relerr_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="^", markersize=12, color="tab:red", label="C-HFD")
ax.errorbar(x_pos, ULH_ccond_relerr_median, yerr=[ULH_ccond_relerr_median-ULH_ccond_relerr_p25,ULH_ccond_relerr_p75-ULH_ccond_relerr_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="*", markersize=15, color="tab:purple", label="U-LH")
ax.errorbar(x_pos.+0.25, CLH_ccond_relerr_median, yerr=[CLH_ccond_relerr_median-CLH_ccond_relerr_p25,CLH_ccond_relerr_p75-CLH_ccond_relerr_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="D", markersize=11, color="tab:orange", label="C-LH")
ax.errorbar(x_pos.+0.5, ACL_ccond_relerr_median, yerr=[ACL_ccond_relerr_median-ACL_ccond_relerr_p25,ACL_ccond_relerr_p75-ACL_ccond_relerr_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="X", markersize=13, color="tab:green", label="ACL")
ax.set_xticks(x_pos)
ax.set_xticklabels([L"$k=3$", L"$k=4$", L"$k=5$", L"$k=6$"], size=28)
ax.set_yticks(range(1.0,1.8,length=5))
ax.set_yticklabels(range(1.0,1.8,length=5), size=18)
ax.tick_params(axis="x", which="both",length=0)
plt.legend(fontsize=18, handletextpad=0.1)
plt.ylabel(L"$\Phi(\hat{C})/\Phi(C)$", fontsize=28)
plt.savefig("cond_edgesize_cond30.pdf", bbox_inches="tight", format="pdf", dpi=400)
