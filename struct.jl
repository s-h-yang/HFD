struct HyperGraph
    incidence::Vector{Vector{Int64}}
    edges::Vector{Vector{Int64}}
    degree::Vector{Int64}
    order::Vector{Int64} # size of hyperedges
    nv::Int64 # number of vertices
    ne::Int64 # number of hyperedges
end
