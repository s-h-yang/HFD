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

####################
# Figure C.1
####################

k = 4
num_of_edges = [950,1000,1050,1110,1180,1250,1330,1410,1500,1600,1720,1850,2000,2200,2500,2800,3200,3750,4500,6000,7500,9500,13000,20000]
num_runs = length(num_of_edges)
true_cond = zeros(Float64, num_runs)
uhfd_cond = zeros(Float64, (50, num_runs))
chfd_cond = zeros(Float64, (50, num_runs))
ulh_cond = zeros(Float64, (50, num_runs))
clh_cond = zeros(Float64, (50, num_runs))
acl_cond = zeros(Float64, (50, num_runs))
uhfd_f1 = zeros(Float64, (50, num_runs))
chfd_f1 = zeros(Float64, (50, num_runs))
ulh_f1 = zeros(Float64, (50, num_runs))
clh_f1 = zeros(Float64, (50, num_runs))
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
deltas = 2.0.^collect(1:12)
L = LH.loss_type(2.0, 0.0)

for num in 1:num_runs

    edges = Set{Set{Int64}}()
    block1 = collect(1:50)
    block2 = collect(51:100)
    while length(edges) < 400
        e = Set(shuffle!(block1)[1:k])
        push!(edges, e)
    end
    while length(edges) < 800
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

    vol_T = sum(H.degree[target])

    true_cond[num] = card_cond(H,collect(1:50))

    for i in 1:num_trials

        @printf("Figure C.1: run %d, trial %d\n", num, i)
        flush(stdout)

        # U-HFD
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = 3*vol_T
        _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=10)
        cluster, _ = uc_sweepcut(H, x./H.degree)
        _, _, uhfd_f1[i,num] = compute_f1(cluster, target)
        uhfd_cond[i,num] = card_cond(H, cluster)

        # C-HFD
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = 3*vol_T
        _, _, x, _ = cardHFD(H, Delta, k, sigma=sigma, max_iters=10)
        cluster, _ = card_sweepcut(H, x./H.degree)
        _, _, chfd_f1[i,num] = compute_f1(cluster, target)
        chfd_cond[i,num] = card_cond(H, cluster)

        # U-LH
        G = LH.graph(M, 1.0)
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        _, _, ulh_f1[i,num] = compute_f1(cluster, target)
        ulh_cond[i,num] = card_cond(H, cluster)

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
        _, _, clh_f1[i,num] = compute_f1(best_cluster, target)
        clh_cond[i,num] = card_cond(H, best_cluster)

        # ACL
        G = LH.graph(M, 1.0)
        Ga, deg = hypergraph_to_bipartite(G)
        kappa = 0.025*ratio
        x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
        x = x[1:size(G.H,2)]
        cluster, _ = uc_sweepcut(H, x./H.degree)
        _, _, acl_f1[i,num] = compute_f1(cluster, target)
        acl_cond[i,num] = card_cond(H, cluster)
    end
end

uhfd_cond_mean = vec(mean(uhfd_cond, dims=1))
chfd_cond_mean = vec(mean(chfd_cond, dims=1))
ulh_cond_mean = vec(mean(ulh_cond, dims=1))
clh_cond_mean = vec(mean(clh_cond, dims=1))
acl_cond_mean = vec(mean(acl_cond, dims=1))
uhfd_cond_std = vec(std(uhfd_cond, dims=1))
chfd_cond_std = vec(std(chfd_cond, dims=1))
ulh_cond_std = vec(std(ulh_cond, dims=1))
clh_cond_std = vec(std(clh_cond, dims=1))
acl_cond_std = vec(std(acl_cond, dims=1))

num = 24
fig,ax = plt.subplots()
plt.grid(linestyle="--", linewidth = 0.5)
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
plt.plot(true_cond[1:num], acl_cond_mean[1:num], color="tab:green", linewidth=5, linestyle=(0,(3,1,1,1,1,1)), label="ACL", alpha=.8)
plt.fill_between(true_cond[1:num], acl_cond_mean[1:num]-.5*acl_cond_std[1:num], acl_cond_mean[1:num]+.5*acl_cond_std[1:num], alpha=.2, color="tab:green")
plt.plot(true_cond[1:num], ulh_cond_mean[1:num], color="tab:purple", linewidth=5, linestyle=(0,(1,1)), label="U-LH", alpha=.8)
plt.fill_between(true_cond[1:num], ulh_cond_mean[1:num]-.5*ulh_cond_std[1:num], ulh_cond_mean[1:num]+.5*ulh_cond_std[1:num], alpha=.2, color="tab:purple")
plt.plot(true_cond[1:num], uhfd_cond_mean[1:num], color="tab:blue", linewidth=5, linestyle="solid", label="U-HFD", alpha=.8)
plt.fill_between(true_cond[1:num], uhfd_cond_mean[1:num]-.5*uhfd_cond_std[1:num], uhfd_cond_mean[1:num]+.5*uhfd_cond_std[1:num], alpha=.2, color="tab:blue")
plt.plot(true_cond[1:num], clh_cond_mean[1:num], color="tab:orange", linewidth=5, linestyle="dashdot", label="C-LH", alpha=.8)
plt.fill_between(true_cond[1:num], clh_cond_mean[1:num]-.5*clh_cond_std[1:num], clh_cond_mean[1:num]+.5*clh_cond_std[1:num], alpha=.2, color="tab:orange")
plt.plot(true_cond[1:num], chfd_cond_mean[1:num], color="tab:red", linewidth=5, linestyle="dashed", label="C-HFD", alpha=.8)
plt.fill_between(true_cond[1:num], chfd_cond_mean[1:num]-.5*chfd_cond_std[1:num], chfd_cond_mean[1:num]+.5*chfd_cond_std[1:num], alpha=.2, color="tab:red")
handles,labels = ax.get_legend_handles_labels()
handles = [handles[3], handles[5], handles[2], handles[4], handles[1]]
labels = [labels[3], labels[5], labels[2], labels[4], labels[1]]
plt.xticks(range(0.04, 0.24, length=6), size=18)
plt.yticks(range(0.04, 0.32, length=8), size=18)
plt.xlabel("Ground-truth conductance", size=30)
plt.ylabel("Output conductance", size=30)
leg = plt.legend(handles,labels,fontsize=18,handletextpad=0.3)
for i in leg.legendHandles
    i.set_linewidth(4)
end
plt.savefig("FigureC1a.pdf", bbox_inches="tight", format="pdf", dpi=400)

uhfd_f1_mean = vec(mean(uhfd_f1, dims=1))
chfd_f1_mean = vec(mean(chfd_f1, dims=1))
ulh_f1_mean = vec(mean(ulh_f1, dims=1))
clh_f1_mean = vec(mean(clh_f1, dims=1))
acl_f1_mean = vec(mean(acl_f1, dims=1))
uhfd_f1_std = vec(std(uhfd_f1, dims=1))
chfd_f1_std = vec(std(chfd_f1, dims=1))
ulh_f1_std = vec(std(ulh_f1, dims=1))
clh_f1_std = vec(std(clh_f1, dims=1))
acl_f1_std = vec(std(acl_f1, dims=1))

num = 24
fig, ax = plt.subplots()
plt.grid(linestyle="--", linewidth = 0.5)
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
plt.plot(true_cond[1:num], acl_f1_mean[1:num], color="tab:green", linewidth=5, linestyle=(0,(3,1,1,1,1,1)), label="ACL", alpha=.8)
plt.fill_between(true_cond[1:num], acl_f1_mean[1:num]-.5*acl_f1_std[1:num], acl_f1_mean[1:num]+.5*acl_f1_std[1:num], alpha=.2, color="tab:green")
plt.plot(true_cond[1:num], ulh_f1_mean[1:num], color="tab:purple", linewidth=5, linestyle=(0,(1,1)), label="U-LH", alpha=.8)
plt.fill_between(true_cond[1:num], ulh_f1_mean[1:num]-.5*ulh_f1_std[1:num], ulh_f1_mean[1:num]+.5*ulh_f1_std[1:num], alpha=.2, color="tab:purple")
plt.plot(true_cond[1:num], uhfd_f1_mean[1:num], color="tab:blue", linewidth=5, linestyle="solid", label="U-HFD", alpha=.8)
plt.fill_between(true_cond[1:num], uhfd_f1_mean[1:num]-.5*uhfd_f1_std[1:num], uhfd_f1_mean[1:num]+.5*uhfd_f1_std[1:num], alpha=.2, color="tab:blue")
plt.plot(true_cond[1:num], chfd_f1_mean[1:num], color="tab:red", linewidth=5, linestyle="dashed", label="C-HFD", alpha=.8)
plt.fill_between(true_cond[1:num], chfd_f1_mean[1:num]-.5*chfd_f1_std[1:num], chfd_f1_mean[1:num]+.5*chfd_f1_std[1:num], alpha=.2, color="tab:red")
plt.plot(true_cond[1:num], clh_f1_mean[1:num], color="tab:orange", linewidth=5, linestyle="dashdot", label="C-LH", alpha=.8)
plt.fill_between(true_cond[1:num], clh_f1_mean[1:num]-.5*clh_f1_std[1:num], clh_f1_mean[1:num]+.5*clh_f1_std[1:num], alpha=.2, color="tab:orange")
handles,labels = ax.get_legend_handles_labels()
handles = [handles[3], handles[4], handles[2], handles[5], handles[1]]
labels = [labels[3], labels[4], labels[2], labels[5], labels[1]]
plt.xticks(range(0.04, 0.24, length=6), size=18)
plt.yticks(range(0.6, 1.00, length=5), size=18)
plt.xlabel("Ground-truth conductance", size=30)
plt.ylabel("Output F1 score", size=30)
leg = plt.legend(handles,labels,fontsize=18,handletextpad=0.3)
for i in leg.legendHandles
    i.set_linewidth(4)
end
plt.savefig("FigureC1b.pdf", bbox_inches="tight", format="pdf", dpi=400)


####################
# Figure C.2
####################

k = 5
num_of_edges = [375,400,425,450,475,500,530,560,600,650,700,750,800,875,950,1050,1200,1400,1600,1900,2300,2800,3300]
num_runs = length(num_of_edges)
true_cond = zeros(Float64, num_runs)
uhfd_cond = zeros(Float64, (50, num_runs))
chfd_cond = zeros(Float64, (50, num_runs))
ulh_cond = zeros(Float64, (50, num_runs))
clh_cond = zeros(Float64, (50, num_runs))
acl_cond = zeros(Float64, (50, num_runs))
uhfd_f1 = zeros(Float64, (50, num_runs))
chfd_f1 = zeros(Float64, (50, num_runs))
ulh_f1 = zeros(Float64, (50, num_runs))
clh_f1 = zeros(Float64, (50, num_runs))
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
deltas = 2.0.^collect(1:12)
L = LH.loss_type(2.0, 0.0)

for num in 1:num_runs

    edges = Set{Set{Int64}}()
    block1 = collect(1:50)
    block2 = collect(51:100)
    while length(edges) < 150
        e = Set(shuffle!(block1)[1:k])
        push!(edges, e)
    end
    while length(edges) < 300
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

    vol_T = sum(H.degree[target])

    true_cond[num] = card_cond(H, collect(1:50))

    for i in 1:num_trials

        @printf("Figure C.2: run %d, trial %d\n", num, i)
        flush(stdout)

        # U-HFD
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = 3*vol_T
        _, _, x, _, _ = ucHFD(H, Delta, sigma=sigma, max_iters=10)
        cluster, _ = uc_sweepcut(H, x./H.degree)
        _, _, uhfd_f1[i,num] = compute_f1(cluster, target)
        uhfd_cond[i,num] = card_cond(H, cluster)

        # C-HFD
        Delta = zeros(Float64, H.nv)
        Delta[seeds[i]] = 3*vol_T
        _, _, x, _ = cardHFD(H, Delta, sigma=sigma, max_iters=10)
        cluster, _ = card_sweepcut(H, x./H.degree)
        _, _, chfd_f1[i,num] = compute_f1(cluster, target)
        chfd_cond[i,num] = card_cond(H, cluster)

        # U-LH
        G = LH.graph(M, 1.0)
        kappa = 0.35*ratio
        x, _, _ = LH.lh_diffusion(G,seeds[i],gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
        _, cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=1)
        _, _, ulh_f1[i,num] = compute_f1(cluster, target)
        ulh_cond[i,num] = card_cond(H, cluster)

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
        _, _, clh_f1[i,num] = compute_f1(best_cluster, target)
        clh_cond[i,num] = card_cond(H, best_cluster)

        # ACL
        G = LH.graph(M, 1.0)
        Ga, deg = hypergraph_to_bipartite(G)
        kappa = 0.025*ratio
        x = PageRank.acl_diffusion(Ga,deg,seeds[i],gamma,kappa)
        x = x[1:size(G.H,2)]
        cluster, _ = uc_sweepcut(H, x./H.degree)
        _, _, acl_f1[i,num] = compute_f1(cluster, target)
        acl_cond[i,num] = card_cond(H, cluster)
    end
end

uhfd_cond_mean = vec(mean(uhfd_cond, dims=1))
chfd_cond_mean = vec(mean(chfd_cond, dims=1))
ulh_cond_mean = vec(mean(ulh_cond, dims=1))
clh_cond_mean = vec(mean(clh_cond, dims=1))
acl_cond_mean = vec(mean(acl_cond, dims=1))
uhfd_cond_std = vec(std(uhfd_cond, dims=1))
chfd_cond_std = vec(std(chfd_cond, dims=1))
ulh_cond_std = vec(std(ulh_cond, dims=1))
clh_cond_std = vec(std(clh_cond, dims=1))
acl_cond_std = vec(std(acl_cond, dims=1))

num = 23
fig,ax = plt.subplots()
plt.grid(linestyle="--", linewidth = 0.5)
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
plt.plot(true_cond[1:num], acl_cond_mean[1:num], color="tab:green", linewidth=5, linestyle=(0,(3,1,1,1,1,1)), label="ACL")
plt.fill_between(true_cond[1:num], acl_cond_mean[1:num]-.5*acl_cond_std[1:num], acl_cond_mean[1:num]+.5*acl_cond_std[1:num], alpha=.2, color="tab:green")
plt.plot(true_cond[1:num], ulh_cond_mean[1:num], color="tab:purple", linewidth=5, linestyle=(0,(1,1)), label="U-LH")
plt.fill_between(true_cond[1:num], ulh_cond_mean[1:num]-.5*ulh_cond_std[1:num], ulh_cond_mean[1:num]+.5*ulh_cond_std[1:num], alpha=.2, color="tab:purple")
plt.plot(true_cond[1:num], uhfd_cond_mean[1:num], color="tab:blue", linewidth=5, linestyle="solid", label="U-HFD")
plt.fill_between(true_cond[1:num], uhfd_cond_mean[1:num]-.5*uhfd_cond_std[1:num], uhfd_cond_mean[1:num]+.5*uhfd_cond_std[1:num], alpha=.2, color="tab:blue")
plt.plot(true_cond[1:num], clh_cond_mean[1:num], color="tab:orange", linewidth=5, linestyle="dashdot", label="C-LH")
plt.fill_between(true_cond[1:num], clh_cond_mean[1:num]-.5*clh_cond_std[1:num], clh_cond_mean[1:num]+.5*clh_cond_std[1:num], alpha=.2, color="tab:orange")
plt.plot(true_cond[1:num], chfd_cond_mean[1:num], color="tab:red", linewidth=5, linestyle="dashed", label="C-HFD")
plt.fill_between(true_cond[1:num], chfd_cond_mean[1:num]-.5*chfd_cond_std[1:num], chfd_cond_mean[1:num]+.5*chfd_cond_std[1:num], alpha=.2, color="tab:red")
handles,labels = ax.get_legend_handles_labels()
handles = [handles[3], handles[5], handles[2], handles[4], handles[1]]
labels = [labels[3], labels[5], labels[2], labels[4], labels[1]]
plt.xticks(range(0.04, 0.18, length=8), size=18)
plt.yticks(range(0.04, 0.28, length=7), size=18)
plt.xlabel("Ground-truth conductance", size=30)
plt.ylabel("Output conductance", size=30)
leg = plt.legend(handles,labels,fontsize=16,handletextpad=0.3)
for i in leg.legendHandles
    i.set_linewidth(3.5)
end
plt.savefig("FigureC2a.pdf", bbox_inches="tight", format="pdf", dpi=400)

uhfd_f1_mean = vec(mean(uhfd_f1, dims=1))
chfd_f1_mean = vec(mean(chfd_f1, dims=1))
ulh_f1_mean = vec(mean(ulh_f1, dims=1))
clh_f1_mean = vec(mean(clh_f1, dims=1))
acl_f1_mean = vec(mean(acl_f1, dims=1))
uhfd_f1_std = vec(std(uhfd_f1, dims=1))
chfd_f1_std = vec(std(chfd_f1, dims=1))
ulh_f1_std = vec(std(ulh_f1, dims=1))
clh_f1_std = vec(std(clh_f1, dims=1))
acl_f1_std = vec(std(acl_f1, dims=1))

num = 23
fig, ax = plt.subplots()
plt.grid(linestyle="--", linewidth = 0.5)
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex = true)
plt.plot(true_cond[1:num], acl_f1_mean[1:num], color="tab:green", linewidth=5, linestyle=(0,(3,1,1,1,1,1)), label="ACL")
plt.fill_between(true_cond[1:num], acl_f1_mean[1:num]-.5*acl_f1_std[1:num], acl_f1_mean[1:num]+.5*acl_f1_std[1:num], alpha=.2, color="tab:green")
plt.plot(true_cond[1:num], ulh_f1_mean[1:num], color="tab:purple", linewidth=5, linestyle=(0,(1,1)), label="U-LH")
plt.fill_between(true_cond[1:num], ulh_f1_mean[1:num]-.5*ulh_f1_std[1:num], min.(ulh_f1_mean[1:num]+.5*ulh_f1_std[1:num],1), alpha=.2, color="tab:purple")
plt.plot(true_cond[1:num], uhfd_f1_mean[1:num], color="tab:blue", linewidth=5, linestyle="solid", label="U-HFD")
plt.fill_between(true_cond[1:num], uhfd_f1_mean[1:num]-.5*uhfd_f1_std[1:num], min.(uhfd_f1_mean[1:num]+.5*uhfd_f1_std[1:num],1), alpha=.2, color="tab:blue")
plt.plot(true_cond[1:num], chfd_f1_mean[1:num], color="tab:red", linewidth=5, linestyle="dashdot", label="C-HFD")
plt.fill_between(true_cond[1:num], chfd_f1_mean[1:num]-.5*chfd_f1_std[1:num], min.(chfd_f1_mean[1:num]+.5*chfd_f1_std[1:num],1), alpha=.2, color="tab:red")
plt.plot(true_cond[1:num], clh_f1_mean[1:num], color="tab:orange", linewidth=5, linestyle="dashed", label="C-LH")
plt.fill_between(true_cond[1:num], clh_f1_mean[1:num]-.5*clh_f1_std[1:num], min.(clh_f1_mean[1:num]+.5*clh_f1_std[1:num],1), alpha=.2, color="tab:orange")
handles,labels = ax.get_legend_handles_labels()
handles = [handles[3], handles[4], handles[2], handles[5], handles[1]]
labels = [labels[3], labels[4], labels[2], labels[5], labels[1]]
plt.xticks(range(0.04, 0.18, length=8), size=18)
plt.yticks(range(0.75, 1.00, length=6), size=18)
plt.xlabel("Ground-truth conductance", size=30)
plt.ylabel("Output F1 score", size=30)
leg = plt.legend(handles,labels,fontsize=16,handletextpad=0.3)
for i in leg.legendHandles
    i.set_linewidth(3.5)
end
plt.savefig("FigureC2b.pdf", bbox_inches="tight", format="pdf", dpi=400)

####################
# Figure C.3
####################


true_ccond = Vector{Float64}()

edges = readdlm("../datasets/synthetic/hyperedges_k3HSBM_cond20.txt", '\t', Int, '\n')
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

    @printf("Figure C.3: k = 3, trial %d\n", i)

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


edges = readdlm("../datasets/synthetic/hyperedges_k4HSBM_cond20.txt", '\t', Int, '\n')
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

    @printf("Figure C.3: k = 4, trial %d\n", i)

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


edges = readdlm("../datasets/synthetic/hyperedges_k5HSBM_cond20.txt", '\t', Int, '\n')
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

    @printf("Figure C.3: k = 5, trial %d\n", i)

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

edges = readdlm("../datasets/synthetic/hyperedges_k6HSBM_cond20.txt", '\t', Int, '\n')
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

    @printf("Figure C.3: k = 6, trial %d\n", i)

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
plt.savefig("FigureC3b.pdf", bbox_inches="tight", format="pdf", dpi=400)

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
ax.set_yticks(range(1.0,2.4,length=8))
ax.set_yticklabels(range(1.0,2.4,length=8), size=18)
ax.tick_params(axis="x", which="both",length=0)
plt.legend(fontsize=18, handletextpad=0.1)
plt.ylabel(L"$\Phi(\hat{C})/\Phi(C)$", fontsize=28)
plt.savefig("FigureC3a.pdf", bbox_inches="tight", format="pdf", dpi=400)



####################
# Figure C.4
####################

true_ccond = Vector{Float64}()

edges = readdlm("../datasets/synthetic/hyperedges_k3HSBM_cond25.txt", '\t', Int, '\n')
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

    @printf("Figure C.4: k = 3, trial %d\n", i)

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


edges = readdlm("../datasets/synthetic/hyperedges_k4HSBM_cond25.txt", '\t', Int, '\n')
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

    @printf("Figure C.4: k = 4, trial %d\n", i)

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


edges = readdlm("../datasets/synthetic/hyperedges_k5HSBM_cond25.txt", '\t', Int, '\n')
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

    @printf("Figure C.4: k = 5, trial %d\n", i)

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

edges = readdlm("../datasets/synthetic/hyperedges_k6HSBM_cond25.txt", '\t', Int, '\n')
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

    @printf("Figure C.4: k = 6, trial %d\n", i)

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
plt.savefig("FigureC4b.pdf", bbox_inches="tight", format="pdf", dpi=400)

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
ax.set_yticks(range(1.0,2.0,length=6))
ax.set_yticklabels(range(1.0,2.0,length=6), size=18)
ax.tick_params(axis="x", which="both",length=0)
plt.legend(fontsize=18, handletextpad=0.1)
plt.ylabel(L"$\Phi(\hat{C})/\Phi(C)$", fontsize=28)
plt.savefig("FigureC4a.pdf", bbox_inches="tight", format="pdf", dpi=400)


####################
# Figure C.5
####################

k = 3
edges = Set{Set{Int64}}()
nodes = collect(1:100)
block1 = collect(1:50)
block2 = collect(51:100)
while length(edges) < 800
    e = Set(shuffle!(block1)[1:k])
    push!(edges, e)
end
while length(edges) < 1600
    e = Set(shuffle!(block2)[1:k])
    push!(edges, e)
end
while length(edges) < 1730
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
nv = length(nodes)
incidence = [Int64[] for v in 1:nv]
for i in 1:ne
    for v in arr[i]
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H1 = HyperGraph(incidence, arr, degree, order, nv, ne)
M1 = hyper2adjmat(H1)
G1 = LH.graph(M1, 1.0)
Ga1, deg1 = hypergraph_to_bipartite(G1)

edges = Set{Set{Int64}}()
nodes = collect(1:100)
block1 = collect(1:50)
block2 = collect(51:100)
while length(edges) < 800
    e = Set(shuffle!(block1)[1:k])
    push!(edges, e)
end
while length(edges) < 1600
    e = Set(shuffle!(block2)[1:k])
    push!(edges, e)
end
while length(edges) < 2900
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
nv = length(nodes)
incidence = [Int64[] for v in 1:nv]
for i in 1:ne
    for v in arr[i]
        push!(incidence[v], i)
    end
end
degree = [length(l) for l in incidence]
order = [length(l) for l in edges]
H2 = HyperGraph(incidence, arr, degree, order, nv, ne)
M2 = hyper2adjmat(H2)
G2 = LH.graph(M2, 1.0)
Ga2, deg2 = hypergraph_to_bipartite(G2)

target = collect(1:50)
L = LH.loss_type(2.0, 0.0)
gamma0 = 0.05 # gamma = 0.05 to match the target conductance
gamma1 = 0.4 # gamma = 0.4 to satisfy the assumption of LH
gamma2 = 0.31 # gamma = 0.3 to match the target conductance
rho = 0.5
x_eps = 1.0e-10
aux_eps = 1.0e-10
max_iters = 100000
ratio = 1/length(target)
kappa_dense = 0.0
kappa_sparse = 0.25*ratio
kappa_dense_acl = 0.00000001
kappa_sparse_acl = 0.1*ratio
sigma = 0.01

target = collect(1:50)
vol_T1 = sum(H1.degree[target])
vol_T2 = sum(H2.degree[target])

cond_hfd1 = zeros(Float64, 50)
cond_lh1_d = zeros(Float64, 50)
cond_lh1_s = zeros(Float64, 50)
cond_acl1_d = zeros(Float64, 50)
cond_acl1_s = zeros(Float64, 50)
f1_hfd1 = zeros(Float64, 50)
f1_lh1_d = zeros(Float64, 50)
f1_lh1_s = zeros(Float64, 50)
f1_acl1_d = zeros(Float64, 50)
f1_acl1_s = zeros(Float64, 50)

cond_hfd2 = zeros(Float64, 50)
cond_lh2_d = zeros(Float64, 50)
cond_lh2_s = zeros(Float64, 50)
cond_acl2_d = zeros(Float64, 50)
cond_acl2_s = zeros(Float64, 50)
f1_hfd2 = zeros(Float64, 50)
f1_lh2_d = zeros(Float64, 50)
f1_lh2_s = zeros(Float64, 50)
f1_acl2_d = zeros(Float64, 50)
f1_acl2_s = zeros(Float64, 50)

for s in 1:50

    Delta = zeros(Float64, H1.nv)
    Delta[s] = 3*vol_T1
    _, _, x, _, _ = ucHFD(H1, Delta, sigma=sigma, max_iters=10)
    cluster, cond_hfd1[s] = uc_sweepcut(H1, x./H1.degree)
    _, _, f1_hfd1[s] = compute_f1(cluster, target)
    Delta = zeros(Float64, H2.nv)
    Delta[s] = 3*vol_T2
    _, _, x, _, _ = ucHFD(H2, Delta, sigma=sigma, max_iters=10)
    cluster, cond_hfd2[s] = uc_sweepcut(H2, x./H2.degree)
    _, _, f1_hfd2[s] = compute_f1(cluster, target)

    x,_,_ = LH.lh_diffusion(G1,[s],gamma1,kappa_dense,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    cond_lh1_d[s], cluster = hyper_sweepcut(G1.H,x,G1.deg,G1.delta,0.0,G1.order,nseeds=1)
    _, _, f1_lh1_d[s] = compute_f1(cluster, target)
    x,_,_ = LH.lh_diffusion(G2,[s],gamma2,kappa_dense,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    cond_lh2_d[s], cluster = hyper_sweepcut(G2.H,x,G2.deg,G2.delta,0.0,G2.order,nseeds=1)
    _, _, f1_lh2_d[s] = compute_f1(cluster, target)
    x,_,_ = LH.lh_diffusion(G1,[s],gamma1,kappa_sparse,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    cond_lh1_s[s], cluster = hyper_sweepcut(G1.H,x,G1.deg,G1.delta,0.0,G1.order,nseeds=1)
    _, _, f1_lh1_s[s] = compute_f1(cluster, target)
    x,_,_ = LH.lh_diffusion(G2,[s],gamma2,kappa_sparse,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    cond_lh2_s[s], cluster = hyper_sweepcut(G2.H,x,G2.deg,G2.delta,0.0,G2.order,nseeds=1)
    _, _, f1_lh2_s[s] = compute_f1(cluster, target)

    x = PageRank.acl_diffusion(Ga1,deg1,[s],gamma0,kappa_dense_acl)
    x ./= deg1
    x = x[1:size(G1.H,2)]
    cond_acl1_d[s], cluster = hyper_sweepcut(G1.H,x,G1.deg,G1.delta,0.0,G1.order,nseeds=1)
    _, _, f1_acl1_d[s] = PRF(target,cluster)
    x = PageRank.acl_diffusion(Ga2,deg2,[s],gamma2,kappa_dense_acl)
    x ./= deg2
    x = x[1:size(G2.H,2)]
    cond_acl2_d[s], cluster = hyper_sweepcut(G2.H,x,G2.deg,G2.delta,0.0,G2.order,nseeds=1)
    _, _, f1_acl2_d[s] = PRF(target,cluster)
    x = PageRank.acl_diffusion(Ga1,deg1,[s],gamma0,kappa_sparse_acl)
    x ./= deg1
    x = x[1:size(G1.H,2)]
    cond_acl1_s[s], cluster = hyper_sweepcut(G1.H,x,G1.deg,G1.delta,0.0,G1.order,nseeds=1)
    _, _, f1_acl1_s[s] = PRF(target,cluster)
    x = PageRank.acl_diffusion(Ga2,deg2,[s],gamma2,kappa_sparse_acl)
    x ./= deg2
    x = x[1:size(G2.H,2)]
    cond_acl2_s[s], cluster = hyper_sweepcut(G2.H,x,G2.deg,G2.delta,0.0,G2.order,nseeds=1)
    _, _, f1_acl2_s[s] = PRF(target,cluster)

end

HFD_f1_median = [median(f1_hfd1),median(f1_hfd2),median(f1_hfd1),median(f1_hfd2)]
LH_f1_median = [median(f1_lh1_d),median(f1_lh2_d),median(f1_lh1_s),median(f1_lh2_s)]
ACL_f1_median = [median(f1_acl1_d),median(f1_acl2_d),median(f1_acl1_s),median(f1_acl2_s)]
HFD_f1_q25 = [quantile(f1_hfd1,0.25),quantile(f1_hfd2,0.25),quantile(f1_hfd1,0.25),quantile(f1_hfd2,0.25)]
HFD_f1_q75 = [quantile(f1_hfd1,0.75),quantile(f1_hfd2,0.75),quantile(f1_hfd1,0.75),quantile(f1_hfd2,0.75)]
LH_f1_q25 = [quantile(f1_lh1_d,0.25),quantile(f1_lh2_d,0.25),quantile(f1_lh1_s,0.25),quantile(f1_lh2_s,0.25)]
LH_f1_q75 = [quantile(f1_lh1_d,0.75),quantile(f1_lh2_d,0.75),quantile(f1_lh1_s,0.75),quantile(f1_lh2_s,0.75)]
ACL_f1_q25 = [quantile(f1_acl1_d,0.25),quantile(f1_acl2_d,0.25),quantile(f1_acl1_s,0.25),quantile(f1_acl2_s,0.25)]
ACL_f1_q75 = [quantile(f1_acl1_d,0.75),quantile(f1_acl2_d,0.75),quantile(f1_acl1_s,0.75),quantile(f1_acl2_s,0.75)]

fig, ax = plt.subplots(figsize=(9,3))
plt.grid(linestyle="--", linewidth = 0.5, axis = "y")
plt.rc("font",family="Times New Roman")
plt.rc("text", usetex=true)
x_pos = [0,3,6,9]
ax.errorbar(x_pos.-0.25, HFD_f1_median, yerr=[HFD_f1_median-HFD_f1_q25,HFD_f1_q75-HFD_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="o", markersize=12, color="tab:blue", label="U-HFD")
ax.errorbar(x_pos, LH_f1_median, yerr=[LH_f1_median-LH_f1_q25,LH_f1_q75-LH_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="*", markersize=15, color="tab:purple", label="U-LH")
ax.errorbar(x_pos.+0.25, ACL_f1_median, yerr=[ACL_f1_median-ACL_f1_q25,ACL_f1_q75-ACL_f1_median], linestyle="", capsize=6, capthick=3, linewidth=4, marker="X", markersize=13, color="tab:green", label="ACL")
ax.set_xticks(x_pos)
ax.set_yticks(range(0.7,1,length=7))
ax.set_xticklabels(["Global sol\nAssum. holds", "Global sol\nAssum. fails", "Localized sol\nAssum. holds", "Localized sol\nAssum. fails"], size=22)
ax.set_yticklabels(["0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"], size=20, fontname="Times New Roman")
ax.tick_params(axis="x", which="both",length=0)
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.spines["left"].set_visible(false)
ax.spines["bottom"].set_visible(false)
plt.ylim([0.695, 1.02])
plt.xlim([-1, 10])
plt.legend(fontsize=18, bbox_to_anchor=(1.02, 1.25), ncol=3)
plt.ylabel("F1 score", fontsize=30, fontname="Times New Roman")
plt.savefig("FigureC5.pdf", bbox_inches="tight", format="pdf", dpi=400)
