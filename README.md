# Orthogonal Neural Networks

This repository contains code for experiments with orthogonal neural networks, which are feed-forward neural networks with
orthogonal transform incorporated into the structure. 

*NOTE*: Code transfer and extension is work in progress, for original refer to our older repository [https://github.com/kzajac97/frequency-supported-neural-networks](https://github.com/kzajac97/frequency-supported-neural-networks).

## Papers

Currently one pre-print (in two versions) is available on arXiv. This code will probably be extended, with follow-up papers.
Table below contains links to papers published with referenced version of our code package, which is `pydentification` (see: [https://github.com/cyber-physical-systems-group/pydentification](https://github.com/cyber-physical-systems-group/pydentification)).

| Paper                                                                       | Link                                      | Version                      |
|-----------------------------------------------------------------------------|-------------------------------------------|------------------------------|
| Orthogonal Transforms in Neural Networks Amount to Effective Regularization | [arXiv](https://arxiv.org/abs/2106.05237) | [v0.1.0-alpha](v0.1.0-alpha) |

## Data

Currently, three datasets were used for experiments:
* Static Affine Benchmark
* Wiener Hammerstein Benchmark
* Silverbox Benchmark

Static Affine Benchmark can be generated using provided notebook (notebooks/static-affine-benchmark-generation.ipynb)
or accessed directly from data directory (data/static-affine-benchmark.csv). Wiener Hammerstein Benchmark and Silverbox
Benchmark (along with many others) can be accessed from https://www.nonlinearbenchmark.org/.