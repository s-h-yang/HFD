using DelimitedFiles

include("../ucHFD.jl")
include("../cardHFD2.jl")
include("../fwHFD2.jl")

# Read data
species = vec(readdlm("../datasets/foodweb/foodweb_species.txt", '\t', String, '\n'))
edges = Vector{Vector{Int64}}()
for line in eachline("../datasets/foodweb/foodweb_edges.txt")
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

# Raptors
println("Query: Raptors")
println("====================")
seednode = 112
Delta = zeros(Float64, H.nv)
Delta[seednode] = 500000
_, _, ex, _, _ = ucHFD(H, Delta, sigma=0.1, max_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Unit cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex, _ = cardHFD(H, Delta, sigma=0.1, max_iters=10, subgrad_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Cardinality cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex, _ = fwHFD(H, Delta, sigma=0.1, max_iters=10)
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
_, _, ex, _, _ = ucHFD(H, Delta, sigma=0.1, max_iters=10, p=2)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Unit cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex = cardHFD(H, Delta, sigma=0.1, max_iters=10, subgrad_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Cardinality-based cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
println()
_, _, ex, _ = fwHFD(H, Delta, sigma=0.1, max_iters=10)
sorted = sortperm(ex./H.degree, rev=true)
filter!(x->x<=122, sorted)
println("Submodular cut-cost top 2 ranked results:")
for i in 2:3
    println(species[sorted[i]])
end
