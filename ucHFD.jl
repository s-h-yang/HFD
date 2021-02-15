using SparseArrays
using LinearAlgebra

include("utils.jl")

function ucHFD(H::HyperGraph, Delta::Vector{Float64}; max_iters::Int64=50,
        sigma::Float64=0.01, p::Real=2, bs_tol::Float64=1.0e-6, r=nothing,
        s=nothing)

    phi = zeros(Float64, H.ne)
    ex = max.(Delta - H.degree, 0.0)
    A = Set{Int}() # maintains active edges
    update_active_set!(H, A, ex)
    nznodes = zeros(Int64,max_iters+2)
    nznodes[1] = count(!iszero, ex)

    if r === nothing || s === nothing
        r = init_r(H)
        s = copy(r)
    else
        fill!(r.nzval, 0.0)
        fill!(s.nzval, 0.0)
    end

    netflow = zeros(Float64, H.nv) # placeholder for netflow
    numverts = zeros(Int64, H.ne) # placeholder for sweepcut

    best_cluster = Vector{Int}()
    best_cond = 1.0

    for k = 1:max_iters
        update_s!(H, r, s, A, ex)
        update_r!(H, r, s, A, sigma, p, phi, bs_tol)
        update_ex!(H, r, A, ex, Delta, netflow)
        update_active_set!(H, A, ex)

        cluster, cond = uc_sweepcut(H, ex./H.degree, numverts=numverts)
        if cond < best_cond
            best_cluster = copy(cluster)
            best_cond = cond
        end
        nznodes[k+1] = count(!iszero, ex)
    end

    nznodes[max_iters+2] = count(!iszero, ex)

    return best_cluster, best_cond, ex, nznodes
end


# r is a |V| x |E| matrix where each column corresponds to a r_e
function init_r(H::HyperGraph)
    colptr = vcat(1, cumsum(H.order).+1)
    rowval = collect(Iterators.flatten(H.edges))
    nzval = zeros(Float64, sum(H.order))
    return SparseMatrixCSC(H.nv, H.ne, colptr, rowval, nzval)
end

# s is a |V| x |E| matrix where each column corresponds to a s_e
function init_s(H::HyperGraph, Delta::Vector{Float64})
    s = init_r(H)
    for v in findall(!iszero, Delta)
        for e in H.incidence[v]
            s[v,e] = max(Delta[v] - H.degree[v], 0) / H.degree[v]
        end
    end
    return s
end

function update_r!(H::HyperGraph, r::SparseMatrixCSC, s::SparseMatrixCSC,
        A::Set{Int}, sigma::Float64, p::Real, phi::Vector{Float64}, bs_tol::Float64)
    for e in A
        if p == 2
            uc_projection!(phi, r, s, e, sigma)
        else
            uc_lp_projection!(phi, r, s, e, sigma, p, bs_tol)
        end
    end
end

function update_ex!(H::HyperGraph, r::SparseMatrixCSC, A::Set{Int},
        ex::Vector{Float64}, Delta::Vector{Float64}, netflow::Vector{Float64})
    fill!(ex, 0.0)
    mul!(netflow, r[:,collect(A)], ones(length(A)))
    for v in 1:H.nv
        ex[v] = max(Delta[v] - netflow[v] - H.degree[v], 0.0)
    end
end

function update_s!(H::HyperGraph, r::SparseMatrixCSC, s::SparseMatrixCSC,
        A::Set{Int}, ex::Vector{Float64})
    for e in A
        for i in nzrange(s, e)
            v = s.rowval[i]
            s.nzval[i] = r.nzval[i] + ex[v]/H.degree[v]
        end
    end
end

function uc_projection!(phi::Vector{Float64}, r::SparseMatrixCSC, s::SparseMatrixCSC, e::Int64, sigma::Float64)
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

function uc_lp_projection!(phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, e::Int64, sigma::Float64, p::Real, bs_tol::Float64)
    q = p / (p - 1)
    indices = nzrange(s, e)
    r.nzval[indices] .= 0.0
    sorted = sortperm(s.nzval[indices], rev=true)
    l = length(indices)
    aq = s.nzval[indices[sorted[1]]]/sigma # aq = |a|^(q-1)*sign(a)
    bq = s.nzval[indices[sorted[l]]]/sigma # bq = |b|^(q-1)*sign(b)
    if aq - bq <= 0
        return
    end
    abq = (abs(aq)^(p-1)*sign(aq) - abs(bq)^(p-1)*sign(bq))^(q-1) # abq = (a-b)^(q-1)
    g = abq
    na = 1 # number of elements in s_e whose value >= sigma*aq
    nb = 1 # number of elements in s_e whose value <= sigma*bq
    wa = 0
    wb = 0
    aq1 = aq
    aq2 = aq
    bq1 = bq
    bq2 = bq
    while true
        wa = na*sigma
        wb = nb*sigma
        aq1 = s.nzval[indices[sorted[na+1]]]/sigma
        bq1 = bq + (aq - aq1)*wa/wb
        bq2 = s.nzval[indices[sorted[l-nb]]]/sigma
        aq2 = aq - (bq2 - bq)*wb/wa
        if bq1 < bq2
            if aq1 <= bq1
                break
            end
            abq1 = (abs(aq1)^(p-1)*sign(aq1) - abs(bq1)^(p-1)*sign(bq1))^(q-1)
            temp = g - abq + abq1 + wa*(aq1 - aq)
        else
            if aq2 <= bq2
                break
            end
            abq2 = (abs(aq2)^(p-1)*sign(aq2) - abs(bq2)^(p-1)*sign(bq2))^(q-1)
            temp = g - abq + abq2 + wa*(aq2 - aq)
        end
        if temp <= 0
            break
        else
            g = temp
            if bq1 < bq2
                aq = aq1
                bq = bq1
                abq = abq1
                na += 1
            else
                aq = aq2
                bq = bq2
                abq = abq2
                nb += 1
                if aq1 == aq2
                    na += 1
                end
            end
        end
    end
    @assert g > 0
    aqL = bq1 < bq2 ? aq1 : aq2
    aqU = aq
    # binary search for the optimal aq in the interval (aqL, aqU)
    while (aqU - aqL > bs_tol) && (aqL < aqL/2 + aqU/2 < aqU)
        aqM = aqL/2 + aqU/2
        bqM = bq + (aq - aqM)*wa/wb
        if aqM < bqM
            aqL = aqM
            continue
        end
        abqM = (abs(aqM)^(p-1)*sign(aqM) - abs(bqM)^(p-1)*sign(bqM))^(q-1)
        temp = g - abq + abqM + wa*(aqM - aq)
        if temp < 0
            aqL = aqM
        else
            aqU = aqM
        end
    end
    aqM = aqU
    bqM = bq + (aq - aqM)*wa/wb
    @assert aqM >= bqM
    for i in indices[sorted[1:na]]
        r.nzval[i] = s.nzval[i] - sigma*aqM
    end
    for i in indices[sorted[(l-nb+1):l]]
        r.nzval[i] = s.nzval[i] - sigma*bqM
    end
end

function update_active_set!(H::HyperGraph, A::Set{Int64}, x::Vector{Float64})
    for v in findall(!iszero, x)
        union!(A, H.incidence[v])
    end
end
