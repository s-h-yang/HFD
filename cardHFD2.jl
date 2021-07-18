using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

function cardHFD(H::HyperGraph, source::Vector{Float64}; sigma::Float64=0.01,
        max_iters::Int64=20, subgrad_iters::Int64=100, r=nothing, s=nothing)

    ex = max.(source - H.degree, 0.0)
    phi = zeros(Float64, H.ne)
    if r === nothing || s === nothing
        r = init_r(H)
        s = copy(r)
    else
        fill!(r.nzval, 0.0)
        fill!(s.nzval, 0.0)
    end
    A = Set{Int}() # maintains active edges
    update_active_set!(H, A, ex)
    netflow = zeros(Float64, H.nv) # pre-allocate vector to store net flows
    edgeC = Dict{Int64,Int64}(e => 0 for e in 1:H.ne) # pre-allocate dict for use in sweepcut

    best_cluster = Vector{Int}()
    best_cond = 1.0

    t1 = time_ns()

    for k = 1:max_iters

        update_s!(H, r, s, A, ex)
        update_r_card!(H, phi, r, s, sigma, A, subgrad_iters=subgrad_iters)
        update_ex!(H, r, A, ex, source, netflow)
        update_active_set!(H, A, ex)

        cluster, cond = card_sweepcut(H, ex./H.degree, edgeC=edgeC)
        if cond < best_cond
            best_cluster = copy(cluster)
            best_cond = cond
        end
        #@printf("Iteration %d, total excess %.2f, active edges %d, excess nodes %d, cond %.4f, objval %.5e\n",
        #    k, sum(ex), length(A), count(!iszero, ex), cond, get_objval(phi, r, s, sigma, A))
    end

    t2 = time_ns()

    return best_cluster, best_cond, ex, (t2-t1)/1e9
end

# r is a |V| x |E| matrix where each column corresponds to a r_e
function init_r(H::HyperGraph)
    colptr = vcat(1, cumsum(H.order).+1)
    rowval = collect(Iterators.flatten(H.edges))
    nzval = zeros(Float64, sum(H.order))
    return SparseMatrixCSC(H.nv, H.ne, colptr, rowval, nzval)
end

# s is a |V| x |E| matrix where each column corresponds to a s_e
function init_s(H::HyperGraph, source::Vector{Float64})
    s = init_r(H)
    for v in findall(!iszero, source)
        for e in H.incidence[v]
            s[v,e] = max(source[v] - H.degree[v], 0) / H.degree[v]
        end
    end
    return s
end

function update_r_card!(H::HyperGraph, phi::Vector{Float64},
        r::SparseMatrixCSC, s::SparseMatrixCSC, sigma::Float64, A::Set{Int};
        subgrad_iters::Int64=100)
    for e in A
        if length(nzrange(s,e)) <= 3
            uc_projection!(phi, r, s, e, sigma)
        else
            card_projection!(phi, r, s, e, sigma, subgrad_iters=subgrad_iters)
        end
    end
end

function update_ex!(H::HyperGraph, r::SparseMatrixCSC, A::Set{Int},
        ex::Vector{Float64}, source::Vector{Float64}, netflow::Vector{Float64})
    fill!(ex, 0.0)
    mul!(netflow, r[:,collect(A)], ones(length(A)))
    for v in 1:H.nv
        ex[v] = max(source[v] - netflow[v] - H.degree[v], 0.0)
    end
end

function update_s!(H::HyperGraph, r::SparseMatrixCSC, s::SparseMatrixCSC,
        A::Set{Int}, ex::Vector{Float64})
    for e in A
        for i in nzrange(s,e)
            v = s.rowval[i]
            s.nzval[i] = r.nzval[i] + ex[v]/H.degree[v]
        end
    end
end

function update_active_set!(H::HyperGraph, A::Set{Int64}, x::Vector{Float64})
    for v in findall(!iszero, x)
        union!(A, H.incidence[v])
    end
end

function uc_projection!(phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, e::Int64, sigma::Float64)
    indices = nzrange(s, e)
    sorted = sortperm(s.nzval[indices], rev=true)
    l = length(indices)
    a = s.nzval[indices[sorted[1]]]/sigma
    b = s.nzval[indices[sorted[l]]]/sigma
    r.nzval[indices] .= 0.0
    g = a - b
    if g <= 0
        phi[e] = 0.0
        return
    end
    na = 1 # number of elements in s_e whose value >= sigma*a
    nb = 1 # number of elements in s_e whose value <= sigma*b
    wa = 0
    wb = 0
    while true
        wa = na*sigma
        wb = nb*sigma
        a1 = s.nzval[indices[sorted[na+1]]]/sigma
        b1 = b + (a - a1)*wa/wb
        b2 = s.nzval[indices[sorted[l-nb]]]/sigma
        a2 = a - (b2 - b)*wb/wa
        if b1 < b2
            temp = g + (a1 - a) - (b1 - b) + wa*(a1 - a)
        else
            temp = g + (a2 - a) - (b2 - b) + wa*(a2 - a)
        end
        if temp <= 0
            break
        else
            g = temp
            if b1 < b2
                a = a1
                b = b1
                na += 1
            else
                a = a2
                b = b2
                nb += 1
                if a1 == a2
                    na += 1
                end
            end
        end
    end
    @assert g > 0
    a -= g*wb/(wa*wb + wa + wb)
    b += g*wa/(wa*wb + wa + wb)
    @assert a >= b
    for i in indices[sorted[1:na]]
        r.nzval[i] = s.nzval[i] - sigma*a
    end
    for i in indices[sorted[(l-nb+1):l]]
        r.nzval[i] = s.nzval[i] - sigma*b
    end
end

function card_projection!(phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, e::Int64, sigma::Float64; subgrad_iters::Int64=100)
    s_e = s.nzval[nzrange(s,e)]
    edgesize = length(s_e)
    x_e = zeros(Float64,edgesize)
    d_e = zeros(Float64,edgesize)
    r_1 = zeros(Float64,edgesize)
    grad = zeros(Float64,edgesize)
    sorted = zeros(Int64,edgesize)
    excess = sum(s_e)
    for k = 1:subgrad_iters
        update_subgrad!(grad, r_1, x_e, s_e, sorted, sigma)
        d_e .= grad
        d_e .*= 1/k/norm(grad)
        x_e .-= d_e
        x_e .+= (excess/sigma - sum(x_e))/length(x_e)
        #@assert abs(sum(x_e)-sum(s_e)/sigma) < 1.0e-3
    end
    r_e = s_e - sigma*x_e
    r.nzval[nzrange(r,e)] = r_e
    phi_sq = dot(r_e,x_e)
    if phi_sq < 0
        phi[e] = 0.0
    else
        phi[e] = sqrt(dot(r_e,x_e))
    end
end

function update_subgrad!(grad::Vector{Float64}, r_1::Vector{Float64},
        x_e::Vector{Float64}, s_e::Vector{Float64}, sorted::Vector{Int64},
        sigma::Float64)
    sortperm!(sorted, x_e, rev=true)
    edgesize = length(sorted)
    ind = floor(Int64,edgesize/2)
    #pos = sorted[1:ind]
    #neg = sorted[(edgesize-ind+1):edgesize]
    #r_1[sorted[1:ind]] .= 1.0/ind
    #r_1[sorted[(edgesize-ind+1):edgesize]] .= -1.0/ind
    for i in sorted[1:ind]
        r_1[i] = 1.0/ind
    end
    for i in sorted[(edgesize-ind+1):edgesize]
        r_1[i] = -1.0/ind
    end
    if edgesize % 2 == 1
        r_1[ind+1] = 0.0
    end
    grad .= dot(r_1,x_e)*r_1 + sigma*x_e - s_e
end

function get_objval(phi, r, s, sigma, A)
    objval = 0.0
    for e in A
        r_e = r.nzval[r.colptr[e]:(r.colptr[e+1]-1)]
        s_e = s.nzval[s.colptr[e]:(s.colptr[e+1]-1)]
        objval = objval + (sigma*phi[e])*phi[e] + norm(s_e-r_e)^2
    end
    return objval
end
