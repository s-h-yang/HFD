# Quick tests

Three quick tests that require minimal dependency and reproduce part of results in the paper:

In your terminal, cd to `tests`,

1) Type: `julia test-trivago-pri-cluster.jl`

   This will run U-HFD and C-HFD in single seed node setting on Cluster PRI
   (Puerto Rico) in the Trivago-clicks hypergraph. It produces F1 scores for
   U-HFD and C-HFD on cluster PRI shown in Table 2.

2) Type: `julia test-foodweb-rank.jl`

   This will produce the node ranking results in Table 3.

3) Type: `julia test-foodweb-high-cluster.jl`

   This will run S-HFD in single seed node setting on Cluster High-level
   Consumers in the Florida-Bay hypergraph. It produces F1 scores for S-HFD
   on Cluster High-level Consumers shown in Table 3.
