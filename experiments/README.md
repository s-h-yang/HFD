# Quick tests

Three quick tests that require minimal dependency and reproduce part of results in the paper can be found in [tests](https://github.com/s-h-yang/HFD/tree/main/tests).

# Complete reproducibility

##### For experiments on synthetic data in the paper:

1) `main-synthetic-unit.jl` produces Figure 4.

2) `main-synthetic-cardinality.jl` produces Figure 5.

##### For experiments on real-world data in the paper:

1) Download the Amazon-reviews dataset at
   https://www.cs.cornell.edu/~arb/data/amazon-reviews/.
   Place the .mat file in the directory `datasets`.

2) To reproduce results on dataset X in the main paper, type:
   `julia main-X-clustering.jl` or `julia main-X-ranking.jl`.

3) When the program terminates, it will have generated several .txt files
   containing raw results for each method. Each line in the .txt file has
   4 numbers, they are precision, recall, F1 and conductance, respectively.
   Taking column-wise median gives the results.

##### For additional experiments in the supplementary material:

1) `supp-synthetic-clustering.jl` produces Figures C.1, C.2, C.3, C.4, C.5.

2) `supp-amazon-clustering.jl` produces intermediate results for Table C.3.

3) `supp-trivago-clustering.jl` produces intermediate results for Table C.4.

4) `supp-school-clustering.jl` produces intermediate results for Table C.5.

5) `supp-mag-clustering.jl` produces intermediate results for Table C.6.

6) `supp-trade-ranking.jl` produces results for Table C.7.

7) `supp-amazon-clustering.jl` produces intermediate results for Table C.8.

8) `supp-foodweb-clustering.jl` produces intermediate results for Table C.9.
