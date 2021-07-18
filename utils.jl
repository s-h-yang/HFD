using SparseArrays
using Combinatorics

include("struct.jl")

function mat2list(H::SparseMatrixCSC)
    m, n = size(H)
    incidence = Vector{Vector{Int64}}()
    edges = Vector{Vector{Int64}}()
    for i = 1:n
        eids = H.rowval[H.colptr[i]:H.colptr[i+1]-1]
        push!(incidence, eids)
    end
    Ht = sparse(H')
    for i = 1:m
        vids = Ht.rowval[Ht.colptr[i]:Ht.colptr[i+1]-1]
        push!(edges, vids)
    end
    return incidence, edges
end

function hyper2adjmat(H::HyperGraph)
    colptr = vcat(1, cumsum(H.degree) .+ 1)
    rowval = collect(Iterators.flatten(H.incidence))
    nzval = ones(Float64, sum(H.degree))
    return SparseMatrixCSC(H.ne, H.nv, colptr, rowval, nzval)
end

function uc_cut(H::HyperGraph, S::Vector{Int64})
    numverts = zeros(Int64, H.ne)
    for v in S
        numverts[H.incidence[v]] .+= 1
    end
    eids = findall(x -> x > 0, numverts)
    return count(!iszero, H.order[eids] - numverts[eids])
end

function uc_cond(H::HyperGraph, S::Vector{Int64})
    vol_H = sum(H.degree)
    vol_S = sum(H.degree[S])
    return uc_cut(H, S) / min(vol_S, vol_H - vol_S)
end

function uc_sweepcut(H::HyperGraph, x::Vector{Float64}; numverts=nothing)
    volH = sum(H.degree)
    volC = 0
    cutC = 0
    if numverts === nothing
        numverts = zeros(Int64, H.ne)
    else
        fill!(numverts, 0)
    end
    C = Vector{Int64}()
    best_cluster = Vector{Int64}()
    best_cond = 1.0
    xnzind = findall(!iszero, x)
    xnzval = x[xnzind]
    sorted_verts = xnzind[sortperm(xnzval, rev = true)]
    for v in sorted_verts
        push!(C, v)
        volC += H.degree[v]
        dcut = 0
        for e in H.incidence[v]
            numverts_prev = numverts[e]
            numverts_curr = numverts_prev + 1
            numverts[e] += 1
            if numverts_prev == 0 && numverts_curr < H.order[e]
                dcut += 1
            elseif numverts_prev > 0 && numverts_curr == H.order[e]
                dcut -= 1
            end
        end
        cutC += dcut
        cond = cutC / min(volC, volH - volC)
        if cond <= best_cond
            best_cluster = copy(C)
            best_cond = cond
        end
    end
    return best_cluster, best_cond
end

function card_cut(H::HyperGraph, S::Vector{Int64})
    edgeC = Dict{Int64,Int64}(e => 0 for e in 1:H.ne)
    for v in S
        for e in H.incidence[v]
            edgeC[e] += 1
        end
    end
    cutC = 0
    for e in 1:H.ne
        edgesize = H.order[e]
        nodecnt = edgeC[e]
        if 0 < nodecnt < edgesize
            cutC += min(nodecnt, edgesize - nodecnt) / floor(edgesize/2)
        end
    end
    return cutC
end

function card_cond(H::HyperGraph, S::Vector{Int64})
    vol_H = sum(H.degree)
    vol_S = sum(H.degree[S])
    return card_cut(H, S) / min(vol_S, vol_H - vol_S)
end

function card_sweepcut(H::HyperGraph, x::Vector{Float64}; edgeC=nothing)
    volH = sum(H.degree)
    volC = 0
    cutC = 0
    if edgeC === nothing
        edgeC = Dict{Int64,Int64}(e => 0 for e in 1:H.ne)
    else
        map!(x->0, values(edgeC))
    end
    C = Vector{Int64}()
    best_cluster = Vector{Int64}()
    best_cond = 1.0
    xnzind = findall(!iszero, x)
    xnzval = x[xnzind]
    sorted_nodes = xnzind[sortperm(xnzval, rev=true)]
    for v in sorted_nodes
        push!(C, v)
        if length(C) == H.nv
            break
        end
        volC += H.degree[v]
        dcut = 0
        for e in H.incidence[v]
            edgesize = H.order[e]
            nodecnt = edgeC[e]
            t = floor(Int64, edgesize/2)
            if nodecnt <= t - 1
                dcut += 1.0/t
            elseif nodecnt >= t + 1
                dcut -= 1.0/t
            elseif (nodecnt == t) && (edgesize % 2 == 0)
                dcut -= 1.0/t
            end
            edgeC[e] += 1
        end
        cutC += dcut
        cond = cutC / min(volC, volH - volC)
        if cond <= best_cond
            best_cluster = copy(C)
            best_cond = cond
        end
    end
    return best_cluster, best_cond
end

function sub_sweepcut(H::HyperGraph, x::Vector{Float64},
        w::Dict{Set{Int64},Float64})
    volH = sum(H.degree)
    volC = 0
    cutC = 0
    edgeC = Dict{Int64,Set{Int64}}(e => Set{Int64}() for e = 1:H.ne)
    C = Vector{Int64}()
    best_cluster = Vector{Int64}()
    best_cond = 1.0
    xnzind = findall(!iszero, x)
    xnzval = x[xnzind]
    sorted_verts = xnzind[sortperm(xnzval, rev=true)]
    for v in sorted_verts
        push!(C, v)
        if length(C) == H.nv
            break
        end
        volC += H.degree[v]
        dcut = 0
        for e in H.incidence[v]
            pos = findall(x -> x == v, H.edges[e])[1]
            prev_w_e = w[edgeC[e]]
            curr_w_e = w[push!(edgeC[e], pos)]
            dcut += (curr_w_e - prev_w_e)
        end
        cutC += dcut
        cond = cutC / min(volC, volH - volC)
        if cond <= best_cond
            best_cluster = copy(C)
            best_cond = cond
        end
    end
    return best_cluster, best_cond
end


function compute_f1(cluster::Vector{Int}, target_cluster::Vector{Int})
    tp = length(intersect(Set(target_cluster), Set(cluster)))
    pr = tp / length(cluster)
    re = tp / length(target_cluster)
    if pr == 0 && re == 0
        f1 = 0
    else
        f1 = 2 * pr * re / (pr + re)
    end
    return pr, re, f1
end

#=
function card_edge_weights(edgesize::Int64)
    if edgesize <= 1
        error("Invalid edge size")
    elseif edgesize == 2
        error("Use unit cut-cost for constant edge size = 2")
    elseif edgesize >= 8
        error("Edge size is not supported")
    else
        e = Set(1:edgesize)
        w = Dict{Set{Int64},Float64}()
        w[Set{Int64}()] = 0.0
        w[e] = 0.0
        if edgesize == 3
            for v = 1:3
                w[Set([v])] = 1
                w[setdiff(e,Set([v]))] = 1
            end
        elseif edgesize == 4
            for v = 1:edgesize
                w[Set([v])] = 1/2
                w[setdiff(e,Set([v]))] = 1/2
            end
            for S in combinations(1:edgesize, 2)
                w[Set(S)] = 1
            end
        elseif edgesize == 5
            for v = 1:edgesize
                w[Set([v])] = 1/2
                w[setdiff(e,Set([v]))] = 1/2
            end
            for S in combinations(1:edgesize, 2)
                w[Set(S)] = 1
                w[setdiff(e,Set(S))] = 1
            end
        elseif edgesize == 6
            for v = 1:edgesize
                w[Set([v])] = 1/3
                w[setdiff(e,Set([v]))] = 1/3
            end
            for S in combinations(1:edgesize, 2)
                w[Set(S)] = 2/3
                w[setdiff(e,Set(S))] = 2/3
            end
            for S in combinations(1:edgesize, 3)
                w[Set(S)] = 1
            end
        elseif edgesize == 7
            for v = 1:edgesize
                w[Set([v])] = 1/3
                w[setdiff(e,Set([v]))] = 1/3
            end
            for S in combinations(1:edgesize, 2)
                w[Set(S)] = 2/3
                w[setdiff(e,Set(S))] = 2/3
            end
            for S in combinations(1:edgesize, 3)
                w[Set(S)] = 1
                w[setdiff(e,Set(S))] = 1
            end
        end
    end
    return w
end
=#
