using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

function fwHFD(H::HyperGraph, Delta::Vector{Float64},
        w::Dict{Set{Int64},Float64}, R::Vector{Vector{Float64}}; p::Real=2,
        sigma::Float64=0.01, max_iters::Int64=20, bs_tol::Float64=1.0e-10)

    ex = max.(Delta - H.degree, 0.0)
    A = Set{Int}() # maintains active edges
    update_active_set!(H, A, ex)
    netflow = zeros(Float64, H.nv) # placeholder for netflow
    phi = zeros(Float64, H.ne)
    r = init_r(H)
    s = copy(r)

    best_cluster = Vector{Int}()
    best_cond = 1.0

    for k = 1:max_iters
        update_s!(H, r, s, A, ex)
        update_r_fw!(H, phi, r, s, sigma, p, A, R, bs_tol)
        update_ex!(H, r, A, ex, Delta, netflow)
        update_active_set!(H, A, ex)

        cluster, cond = sub_sweepcut(H, ex./H.degree, w)
        if cond < best_cond
            best_cluster = copy(cluster)
            best_cond = cond
        end
    end

    return best_cluster, best_cond, ex
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

function update_r_fw!(H::HyperGraph, phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, sigma::Float64, p::Real, A::Set{Int},
        R::Vector{Vector{Float64}}, bs_tol::Float64)
    s_e = zeros(Float64, 4)
    r_1 = zeros(Float64, 4)
    r_e_opt = zeros(Float64, 4)
    r_e_temp = zeros(Float64, 4)
    a = zeros(Float64, 4)
    for e in A
        sub_projection_fw!(phi, r, s, e, sigma, p, R, bs_tol, s_e, r_1,
            r_e_opt, r_e_temp, a)
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

function update_active_set!(H::HyperGraph, A::Set{Int64}, x::Vector{Float64})
    for v in findall(!iszero, x)
        union!(A, H.incidence[v])
    end
end

function sub_projection_fw!(phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, e::Int64, sigma::Float64, p::Real,
        R::Vector{Vector{Float64}}, bs_tol::Float64, s_e::Vector{Float64},
        r_1::Vector{Float64}, r_e_opt::Vector{Float64},
        r_e_temp::Vector{Float64}, a::Vector{Float64})
    phi_0 = phi[e]
    s_e .= s.nzval[nzrange(s,e)]
    r_1 .= R[1]
    r_e_opt .= r_1
    phi_e_opt = compute_phi!(r_1, s_e, sigma, p, phi_0, bs_tol, a)
    r_e_opt .*= phi_e_opt
    fval_opt = get_sub_fval!(phi_e_opt, r_e_opt, s_e, sigma, p, a)
    for i = 2:length(R)
        r_1 .= R[i]
        r_e_temp .= r_1
        phi_e_temp = compute_phi!(r_1, s_e, sigma, p, phi_0, bs_tol, a)
        r_e_temp .*= phi_e_temp
        fval_temp = get_sub_fval!(phi_e_temp, r_e_temp, s_e, sigma, p, a)
        if fval_temp < fval_opt
            phi_e_opt = phi_e_temp
            r_e_opt .= r_e_temp
            fval_opt = fval_temp
        end
    end
    phi[e] = phi_e_opt
    r.nzval[nzrange(r,e)] .= r_e_opt
end

function compute_phi!(r_1, s_e, sigma, p, phi_0, bs_tol, a)
    if p == 2
        return dot(r_1,s_e)/(sigma + dot(r_1,r_1))
    end
    if get_grad!(phi_0, r_1, s_e, sigma, p, a) < 0
        L = phi_0
        U = L + 1
        t = 1
        while get_grad!(U, r_1, s_e, sigma, p, a) < 0 && t <= 100
            L = U
            U = L + 2^t
            t += 1
        end
        if t >= 100
            @warn "Binary search for phi_e failed!"
        end
    else
        U = phi_0
        L = U - 1
        t = 1
        while get_grad!(L, r_1, s_e, sigma, p, a) > 0 && t <= 100
            U = L
            L = U - 2^t
            t += 1
        end
        if t >= 100
            @warn "Binary search for phi_e failed!"
        end
    end
    if L < 0
        L = 0.0
    end
    if U <= 0
        return 0.0
    end
    @assert U > L
    while (U - L > bs_tol) && (L < L/2 + U/2 < U)
        M = L/2 + U/2
        if get_grad!(M, r_1, s_e, sigma, p, a) > 0
            U = M
        else
            L = M
        end
    end
    return L/2 + U/2
end

function get_grad!(phi_e, r_1, s_e, sigma, p, a)
    a .= r_1
    a .*= phi_e
    a .-= s_e
    a .^= (p-1)
    return (sigma*abs(phi_e))^(p-1)*sign(phi_e) + dot(r_1, a)
end

function get_sub_fval!(phi_e, r_e, s_e, sigma, p, a)
    a .= s_e
    a .-= r_e
    return (sigma*phi_e)^(p-1)*phi_e + norm(a,p)^p
end


function get_objval(phi, r, s, sigma, p, A)
    objval = 0.0
    for e in A
        r_e = r.nzval[r.colptr[e]:(r.colptr[e+1]-1)]
        s_e = s.nzval[s.colptr[e]:(s.colptr[e+1]-1)]
        objval = objval + (sigma*phi[e])^(p-1)*phi[e] + norm(s_e-r_e,p)^p
    end
    return objval
end
