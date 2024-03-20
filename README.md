# Orthogonal Neural Networks

This repository contains code for experiments with orthogonal neural networks, which are feed-forward neural networks with
orthogonal transform incorporated into the structure.

## Papers

Currently one pre-print (in two versions) is available on arXiv. This code will probably be extended, with follow-up papers.
Table below contains links to papers published with referenced version of our code package, which is `pydentification` (see: [https://github.com/cyber-physical-systems-group/pydentification](https://github.com/cyber-physical-systems-group/pydentification)).

*Note*: The code was transferred after writing the paper, all experiments can be run using code from [https://github.com/kzajac97/frequency-supported-neural-networks](https://github.com/kzajac97/frequency-supported-neural-networks).
Changes added later are aligning with the `pydentification` library, which we share for all our research, and it contains
transferred functionalities from the old repository.

| Version                                                                                                                             | Paper                                                                                                                   | Description                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [https://github.com/kzajac97/frequency-supported-neural-networks](https://github.com/kzajac97/frequency-supported-neural-networks)  | [Orthogonal Transforms in Neural Networks Amount to Effective Regularization (V1)](https://arxiv.org/abs/2305.06344v1)  | Early version of the pre-print, uses different repository          |
| v1.0-alpha                                                                                                                          | [Orthogonal Transforms in Neural Networks Amount to Effective Regularization (V2)](https://arxiv.org/abs/2305.06344)    | Current version of the code, uses this repository and dependencies |

## Dependencies

We based our implementation on our core library, `pydentification`  (see: https://github.com/cyber-physical-systems-group/pydentification).
Most of the feature is implemented in [`v0.1.0-alpha`](https://github.com/cyber-physical-systems-group/pydentification/releases/tag/v0.1.0-alpha) 
and the [`v0.3.0`](https://github.com/cyber-physical-systems-group/pydentification/releases/tag/v0.3.0) version contains
the code for running the experiments (entrypoints etc.), experimentation code was implemented here and generalized and
moved to main library.

## Data

Currently, three datasets were used for experiments:
* Static Affine Benchmark
* Wiener Hammerstein Benchmark
* Silverbox Benchmark

Static Affine Benchmark can be generated using provided notebook (`notebooks/static-affine-benchmark-generation.ipynb`)
or accessed directly from data directory (`data/static-affine-benchmark.csv`). Wiener Hammerstein Benchmark and Silverbox
Benchmark (along with many others) can be accessed from https://www.nonlinearbenchmark.org/.