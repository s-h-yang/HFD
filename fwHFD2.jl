using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

function fwHFD(H::HyperGraph, source::Vector{Float64};
        sigma::Float64=0.01, max_iters::Int64=20, bs_tol::Float64=1.0e-10)

    ex = max.(source - H.degree, 0.0)
    phi = zeros(Float64, H.ne)
    A = Set{Int}() # maintains active edges
    update_active_set!(H, A, ex)
    netflow = zeros(Float64, H.nv) # pre-allocate vector to store net flows
    r = init_r(H)
    s = copy(r)

    best_cluster = Vector{Int}()
    best_cond = 1.0

    # define the submodular weight
    w = Dict{Set{Int64},Float64}()
    w[Set{Int64}()] = 0.0
    w[Set(1:4)] = 0.0
    for i in 1:4
        w[Set([i])] = 0.5
        w[setdiff(Set(1:4),Set([i]))] = 0.5
    end
    w[Set([1,2])] = 0.0
    w[Set([3,4])] = 0.0
    w[Set([1,3])] = 1.0
    w[Set([2,4])] = 1.0
    w[Set([1,4])] = 1.0
    w[Set([2,3])] = 1.0

    t1 = time_ns()

    for k = 1:max_iters
        update_s!(H, r, s, A, ex)
        update_r_fw!(H, phi, r, s, sigma, A)
        update_ex!(H, r, A, ex, source, netflow)
        update_active_set!(H, A, ex)

        cluster, cond = sub_sweepcut(H, ex./H.degree, w)
        if cond < best_cond
            best_cluster = copy(cluster)
            best_cond = cond
        end
        #@printf("Iteration %d, total excess %.2f, active edges %d, excess nodes %d, cond %.4f, objval %.5e\n",
        #k, sum(ex), length(A), count(!iszero, ex), cond, get_objval(phi, r, s, sigma, A))
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

function update_r_fw!(H::HyperGraph, phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, sigma::Float64, A::Set{Int})
    s_e = zeros(Float64, 4)
    r_e_opt = zeros(Float64, 4)
    r_e_temp = zeros(Float64, 4)
    temp = zeros(Float64, 4)
    for e in A
        fw_projection!(phi, r, s, e, sigma, s_e, r_e_opt, r_e_temp, temp)
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

function fw_projection!(phi::Vector{Float64}, r::SparseMatrixCSC,
        s::SparseMatrixCSC, e::Int64, sigma::Float64,
        s_e::Vector{Float64}, r_e_opt::Vector{Float64},
        r_e_temp::Vector{Float64}, temp::Vector{Float64})
    s_e .= s.nzval[nzrange(s,e)]
    d12 = s_e[1] - s_e[2]
    d34 = s_e[3] - s_e[4]
    if abs(d12) < 1.0e-8 && abs(d34) < 1.0e-8
        r_e_opt .= 0.0
        phi_e_opt = 0.0
    elseif abs(d12) < 1.0e-8
        r_e_opt[1] = 0.0
        r_e_opt[2] = 0.0
        if d34 > 0
            r_e_opt[3] = 0.5
            r_e_opt[4] = -0.5
            phi_e_opt = dot(r_e_opt,s_e)/(sigma + dot(r_e_opt,r_e_opt))
            r_e_opt .*= phi_e_opt
        else
            r_e_opt[3] = -0.5
            r_e_opt[4] = 0.5
            phi_e_opt = dot(r_e_opt,s_e)/(sigma + dot(r_e_opt,r_e_opt))
            r_e_opt .*= phi_e_opt
        end
    elseif abs(d34) < 1.0e-8
        r_e_opt[3] = 0.0
        r_e_opt[4] = 0.0
        if d12 > 0
            r_e_opt[1] = 0.5
            r_e_opt[2] = -0.5
            phi_e_opt = dot(r_e_opt,s_e)/(sigma + dot(r_e_opt,r_e_opt))
            r_e_opt .*= phi_e_opt
        else
            r_e_opt[1] = -0.5
            r_e_opt[2] = 0.5
            phi_e_opt = dot(r_e_opt,s_e)/(sigma + dot(r_e_opt,r_e_opt))
            r_e_opt .*= phi_e_opt
        end
    else
        if d12 > 0
            r_e_opt[1] = 0.5
            r_e_opt[2] = -0.5
        else
            r_e_opt[1] = -0.5
            r_e_opt[2] = 0.5
        end
        if d34 > 0
            r_e_opt[3] = 0.5
            r_e_opt[4] = -0.5
        else
            r_e_opt[3] = -0.5
            r_e_opt[4] = 0.5
        end
        phi_e_opt = dot(r_e_opt,s_e)/(sigma + dot(r_e_opt,r_e_opt))
        r_e_opt .*= phi_e_opt
        fval_opt = get_sub_fval!(phi_e_opt, r_e_opt, s_e, sigma, temp)
        a = (0.5 + sigma)*abs(d12/d34)
        b = (0.5 + sigma)*abs(d34/d12)
        if a < 0.5
            if d12 > 0
                r_e_temp[1] = a
                r_e_temp[2] = -a
            else
                r_e_temp[1] = -a
                r_e_temp[2] = a
            end
            if d34 > 0
                r_e_temp[3] = 0.5
                r_e_temp[4] = -0.5
            else
                r_e_temp[3] = -0.5
                r_e_temp[4] = 0.5
            end
            phi_e_temp = dot(r_e_temp,s_e)/(sigma + dot(r_e_temp,r_e_temp))
            r_e_temp .*= phi_e_temp
            fval_temp = get_sub_fval!(phi_e_temp, r_e_temp, s_e, sigma, temp)
            if fval_temp < fval_opt
                phi_e_opt = phi_e_temp
                r_e_opt .= r_e_temp
                fval_opt = fval_temp
            end
        end
        if b < 0.5
            if d12 > 0
                r_e_temp[1] = 0.5
                r_e_temp[2] = -0.5
            else
                r_e_temp[1] = -0.5
                r_e_temp[2] = 0.5
            end
            if d34 > 0
                r_e_temp[3] = b
                r_e_temp[4] = -b
            else
                r_e_temp[3] = -b
                r_e_temp[4] = b
            end
            phi_e_temp = dot(r_e_temp,s_e)/(sigma + dot(r_e_temp,r_e_temp))
            r_e_temp .*= phi_e_temp
            fval_temp = get_sub_fval!(phi_e_temp, r_e_temp, s_e, sigma, temp)
            if fval_temp < fval_opt
                phi_e_opt = phi_e_temp
                r_e_opt .= r_e_temp
                fval_opt = fval_temp
            end
        end
    end
    phi[e] = phi_e_opt
    r.nzval[nzrange(r,e)] .= r_e_opt
end

function get_sub_fval!(phi_e, r_e, s_e, sigma, temp)
    temp .= s_e
    temp .-= r_e
    return sigma*phi_e*phi_e + dot(temp,temp)
end


function get_objval(phi, r, s, sigma, A)
    objval = 0.0
    for e in A
        r_e = r.nzval[r.colptr[e]:(r.colptr[e+1]-1)]
        s_e = s.nzval[s.colptr[e]:(s.colptr[e+1]-1)]
        objval = objval + sigma*(phi[e]^2) + norm(s_e-r_e)^2
    end
    return objval
end
