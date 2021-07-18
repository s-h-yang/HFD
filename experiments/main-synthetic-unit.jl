using Random
using Statistics
using Printf

using PyCall, PyPlot
plt = pyimport("matplotlib.pyplot")

include("../utils.jl");
include("../ucHFD.jl")
include("../LHQD/common.jl")
include("../LHQD/local-hyper.jl")
include("../LHQD/PageRank.jl")

k = 3
num_of_edges = [3500,3600,3700,3800,3900,4000,4100,4200,4300,4500,4700,4900,
    5100,5300,5600,5900,6200,6500,6800,7100,7400,7700,8000,8500,9000,10000,
    11000,12000]
num_runs = length(num_of_edges)
true_cond = zeros(Float64, num_runs)
uhfd_cond = zeros(Float64, (50, num_runs))
ulh_cond = zeros(Float64, (50, num_runs))
acl_cond = zeros(Float64, (50, num_runs))
uhfd_f1 = zeros(Float64, (50, num_runs))
ulh_f1 = zeros(Float64, (50, num_runs))
acl_f1 = zeros(Float64, (50, num_runs))

target = collect(1:50)
seeds = copy(target)
num_trials = length(seeds)
sigma = 0.01
gamma = 0.1
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-16
max_iters = 1000000
ratio = 1/length(target)
L = LH.loss_type(2.0, 0.0)

for num in 1:num_runs

    # generate hypergraph
    edges = Set{Set{Int64}}()
    block1 = collect(1:50)
    block2 = collect(51:100)
    while length(edges) < 1500
        e = Set(shuffle!(block1)[1:k])
        push!(edges, e)
    end
    while length(edges) < 3000
        e = Set(shuffle!(block2)[1:k])
        push!(edges, e)
    end
    while length(edges) < num_of_edges[num]
        if rand() < 0.5
            e = Set(shuffle!(block1)[1:k-1])
            push!(e, rand(block2))
        else
            e = Set(shuffle!(block2)[1:k-1])
            push!(e, rand(block1))
        end
        push!(edges, e)
    end
    arr = []
    for e in edges
        push!(arr, collect(e))
    end
    ne = length(arr)
    nv = 100
    incidence = [Int64[] for v in 1:nv]
    for i in 1:ne
        for v in arr[i]
            push!(incidence[v], i)
        end
    end
    degree = [length(l) for l in incidence]
    order = [length(l) for l in edges]
    H = HyperGraph(incidence, arr, degree, order, nv, ne)
    M = hyper2adjmat(H)
    G = LH.graph(M, 1.0)
    Ga, deg = hypergraph_to_bipartite(G)

    vol_T = sum(H.degree[target])

    true_cond[num] = uc_cond(H,collect(1:50))

    for i in 1:num_trials

        @printf("Run %d, trial %d\n", num, i)
        flush(stdout)

        # U-HFD
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = 3*vol_T
        _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=10)
        cluster, _ = uc_sweepcut(H, x./H.degree)
        _, _, uhfd_f1[i,num] = compute_f1(cluster, target)
        uhfd_cond[i,num] = uc_cond(H, cluster)

        # U-LH
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,
            max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        _, _, ulh_f1[i,num] = compute_f1(cluster, target)
        ulh_cond[i,num] = uc_cond(H, cluster)

        # ACL
        kappa = 0.025*ratio
        x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
        x = x[1:size(G.H,2)]
        cluster, _ = uc_sweepcut(H, x./H.degree)
        _, _, acl_f1[i,num] = compute_f1(cluster, target)
        acl_cond[i,num] = uc_cond(H, cluster)
    end
end

uhfd_cond_mean = vec(mean(uhfd_cond, dims=1))
ulh_cond_mean = vec(mean(ulh_cond, dims=1))
acl_cond_mean = vec(mean(acl_cond, dims=1))
uhfd_cond_std = vec(std(uhfd_cond, dims=1))
ulh_cond_std = vec(std(ulh_cond, dims=1))
acl_cond_std = vec(std(acl_cond, dims=1))

num = 28
fig,ax = plt.subplots()
plt.grid(linestyle="--", linewidth = 0.5)
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
plt.fill_between(true_cond[1:num], uhfd_cond_mean[1:num]-uhfd_cond_std[1:num],
    uhfd_cond_mean[1:num]+uhfd_cond_std[1:num], alpha=.25, color="tab:blue")
plt.fill_between(true_cond[1:num], ulh_cond_mean[1:num]-ulh_cond_std[1:num],
    ulh_cond_mean[1:num]+ulh_cond_std[1:num], alpha=.25, color="tab:purple")
plt.fill_between(true_cond[1:num], acl_cond_mean[1:num]-acl_cond_std[1:num],
    acl_cond_mean[1:num]+acl_cond_std[1:num], alpha=.25, color="tab:green")
plt.plot(true_cond[1:num], uhfd_cond_mean[1:num], color="tab:blue",
    linewidth=5, linestyle="solid", label="U-HFD", alpha=1)
plt.plot(true_cond[1:num], acl_cond_mean[1:num], color="tab:green",
    linewidth=5, linestyle=(0,(3,1,1,1,1,1)), label="ACL", alpha=1)
plt.plot(true_cond[1:num], ulh_cond_mean[1:num], color="tab:purple",
    linewidth=5, linestyle=(0,(1,1)), label="U-LH", alpha=1)
handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[3], handles[2]]
labels = [labels[1], labels[3], labels[2]]
plt.xticks(range(0.1, 0.5, length=5), size=20)
plt.yticks(range(0.1, 0.5, length=5), size=20)
plt.xlabel("Ground-truth conductance", size=30)
plt.ylabel("Output conductance", size=30)
leg = plt.legend(handles,labels,fontsize=18,handletextpad=.5,handlelength=2.5)
for i in leg.legendHandles
    i.set_linewidth(4)
end
plt.savefig("unit_cond.pdf", bbox_inches="tight", format="pdf", dpi=400)

uhfd_f1_mean = vec(mean(uhfd_f1, dims=1))
ulh_f1_mean = vec(mean(ulh_f1, dims=1))
acl_f1_mean = vec(mean(acl_f1, dims=1))
uhfd_f1_std = vec(std(uhfd_f1, dims=1))
ulh_f1_std = vec(std(ulh_f1, dims=1))
acl_f1_std = vec(std(acl_f1, dims=1))

num = 28
fig, ax = plt.subplots()
plt.grid(linestyle="--", linewidth = 0.5)
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
plt.fill_between(true_cond[1:num], uhfd_f1_mean[1:num]-uhfd_f1_std[1:num],
    min.(uhfd_f1_mean[1:num]+uhfd_f1_std[1:num],1.0), alpha=.25,
    color="tab:blue")
plt.fill_between(true_cond[1:num], ulh_f1_mean[1:num]-ulh_f1_std[1:num],
    ulh_f1_mean[1:num]+ulh_f1_std[1:num], alpha=.25, color="tab:purple")
plt.fill_between(true_cond[1:num], acl_f1_mean[1:num]-acl_f1_std[1:num],
    acl_f1_mean[1:num]+acl_f1_std[1:num], alpha=.25, color="tab:green")
plt.plot(true_cond[1:num], uhfd_f1_mean[1:num], color="tab:blue",
    linewidth=5, linestyle="solid", label="U-HFD")
plt.plot(true_cond[1:num], acl_f1_mean[1:num], color="tab:green",
    linewidth=5, linestyle=(0,(3,1,1,1,1,1)), label="ACL")
plt.plot(true_cond[1:num], ulh_f1_mean[1:num], color="tab:purple",
    linewidth=5, linestyle=(0,(1,1)), label="U-LH")
handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[3], handles[2]]
labels = [labels[1], labels[3], labels[2]]
plt.xticks(range(0.1, 0.5, length=5), size=20)
plt.yticks(range(0.5, 1.0, length=6), size=20)
plt.xlabel("Ground-truth conductance", size=30)
plt.ylabel("Output F1 score", size=30)
leg = plt.legend(handles,labels,fontsize=18,handletextpad=.5,handlelength=2.5)
for i in leg.legendHandles
    i.set_linewidth(4)
end
plt.savefig("unit_f1.pdf", bbox_inches="tight", format="pdf", dpi=400)
