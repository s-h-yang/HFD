# Quick tests

Three quick tests that require minimal dependency and reproduce part of results in the paper.

In your terminal

1) Type: `julia test-trivago-iceland.jl`.
This will run HFD in single seed node setting on Cluster Iceland in the Trivago-clicks hypergraph. Results are shown in Table 4.

2) Type: `julia test-school-class7.jl`.
This will run HFD in single seed node setting on Class 7 in the High-school-contact hypergraph. Results are shown in Table 4.

3) Type: `julia foodweb-ranking.jl`.
This will reproduce the search ranking results shown in Table 1.


# Complete reproducibility

1) Please download the Amazon-reviews dataset at https://www.cs.cornell.edu/~arb/data/amazon-reviews/ and place the .mat file under datasets/

2) To reproduce results on dataset X in the main paper, type: `julia X-experiments-main.jl`

3) To reproduce results on dataset X in the appendix, type: `julia X-experiments-supp.jl`

4) When the program terminates, it will have generated several .txt files containing raw results for each method. Each line in the .txt file has 4 numbers, they are precision, recall, F1, conductance. Taking column-wise median gives the results in Table 4
