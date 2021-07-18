include("../ucHFD.jl")
include("../cardHFD2.jl")
include("../fwHFD2.jl")

# Read data
countries = Vector{String}()
for line in eachline("../datasets/oil-trade/countries-2017.txt")
    push!(countries, line)
end
edges = Vector{Vector{Int64}}()
for line in eachline("../datasets/oil-trade/hyperedges-oil-trade-2017.txt")
    push!(edges, parse.(Int64, split(line,",")))
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

# Iran
println("Query: Iran")
println("====================")
seednode = 80
Delta = zeros(Float64, H.nv)
Delta[seednode] = 500000
_, _, ex, _, _ = ucHFD(H, Delta, sigma=0.1, max_iters=100)
scores = [d == 0 ? 0.0 : x/d for (x,d) in zip(ex,H.degree)]
sorted = sortperm(scores, rev=true)
println("Unit cut-cost top 2 ranked results:")
for i in 2:3
    println(countries[sorted[i]])
end
println()
_, _, ex, _ = cardHFD(H, Delta, sigma=0.1, max_iters=20, subgrad_iters=10)
scores = [d == 0 ? 0.0 : x/d for (x,d) in zip(ex,H.degree)]
sorted = sortperm(scores, rev=true)
println("Cardinality cut-cost top 2 ranked results:")
for i in 2:3
    println(countries[sorted[i]])
end
println()
_, _, ex, _ = fwHFD(H, Delta, sigma=0.1, max_iters=20)
scores = [d == 0 ? 0.0 : x/d for (x,d) in zip(ex,H.degree)]
sorted = sortperm(scores, rev=true)
println("Submodular cut-cost top 2 ranked results:")
for i in 2:3
    println(countries[sorted[i]])
end
