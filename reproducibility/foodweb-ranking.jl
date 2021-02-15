using DelimitedFiles
using Printf

include("../utils.jl")
include("../ucHFD.jl")
include("../fwHFD.jl")

# Read data
species = vec(readdlm("datasets/foodweb/foodweb_species.txt", '\t', String, '\n'))
edges = Vector{Vector{Int64}}()
for line in eachline("datasets/foodweb/foodweb_edges.txt")
    push!(edges, parse.(Int64, split(line,"\t")))
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

# weights
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
R1 = [[0.5,0.5,-0.5,-0.5],
      [0.5,-0.5,0.5,-0.5],
      [0.5,-0.5,-0.5,0.5],
      [-0.5,0.5,0.5,-0.5],
      [-0.5,0.5,-0.5,0.5],
      [-0.5,-0.5,0.5,0.5],
      [0.5,-0.5,0.0,0.0],
      [0.5,0.0,-0.5,0.0],
      [0.5,0.0,0.0,-0.5],
      [0.0,0.5,-0.5,0.0],
      [0.0,0.5,0.0,-0.5],
      [0.0,0.0,0.5,-0.5],
      [-0.5,0.5,0.0,0.0],
      [-0.5,0.0,0.5,0.0],
      [-0.5,0.0,0.0,0.5],
      [0.0,-0.5,0.5,0.0],
      [0.0,-0.5,0.0,0.5],
      [0.0,0.0,-0.5,0.5]]
R2 = [[0.5,-0.5,0.5,-0.5],
      [0.5,-0.5,-0.5,0.5],
      [-0.5,0.5,0.5,-0.5],
      [-0.5,0.5,-0.5,0.5],
      [0.5,-0.5,0.0,0.0],
      [-0.5,0.5,0.0,0.0],
      [0.0,0.0,0.5,-0.5],
      [0.0,0.0,-0.5,0.5]]

# Raptors
println("Query: Raptors")
println("====================")
seednode = 112
Delta = zeros(Float64, H.nv)
Delta[seednode] = 500000
_, _, ex, _ = ucHFD(H, Delta, sigma=0.1, max_iters=10, p=2)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Unit cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex = fwHFD(H, Delta, w, R1, p=2, sigma=0.1, max_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Cardinality-based cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex = fwHFD(H, Delta, w, R2, p=2, sigma=0.1, max_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Submodular cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end

# Gray Snapper
println()
println()
println("Query: Gray Snapper")
println("====================")
seednode = 80
Delta = zeros(Float64, H.nv)
Delta[seednode] = 500000
_, _, ex, _ = ucHFD(H, Delta, sigma=0.1, max_iters=10, p=2)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Unit cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex = fwHFD(H, Delta, w, R1, p=2, sigma=0.1, max_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Cardinality-based cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex = fwHFD(H, Delta, w, R2, p=2, sigma=0.1, max_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Submodular cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
