# General Graph Random Features

This repository accompanies the paper 'General Graph Random Features', which was published at ICLR 2024 by Isaac Reid, Krzysztof Choromanski, Eli Berger and Adrian Weller. The manuscript can be found [here](https://arxiv.org/abs/2310.04859). 
General graph random features (GRFs) can use random walks to construct sparse Monte Carlo estimates of graph node kernels -- or, more specifically, functions of a weighted adjacency matrix.

**This repo.** 
This lightweight repo reproduces Fig. 2 of the paper, which shows the kernel approximation quality against the number of random walkers.
We consider a range of kernels on a range of real and synthetic graphs. Naturally, as the number of samples grow the approximation error drops. The estimator is unbiased.[^1]

<div align="center">
  <img src="/grfs_schematic.png" alt="Alt text" width="800">
</div>

**Installation instructions.** 
The requirements of the repo are minimal.
For a quick installation, run:

```bash
conda env create -f environment.yml --name new_environment_name
```
in the downloaded repo in terminal. But in practice one only needs numpy, matplotlib, scipy and ipykernel.

**Significance and extensions.**
Brute force methods typically incur cubic time complexity in the number of graph nodes, whereas GRFs are linear. 
We provide exponential concentration results in [this follow-up paper](https://arxiv.org/abs/2410.03462).
GRFs unlock scalable implicit kernel learning in feature space, which have proved a powerful technique for injecting information about graph structure into Transformers -- a so-called 'topological inductive bias'.

[^1]: Really, for unbiased estimates of diagonal kernel entries, one ought to sample two sets of independent walks to construct two independent features.
But this technical detail makes little difference in practice so we omit further discussion.

