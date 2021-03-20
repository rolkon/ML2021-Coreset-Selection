# Core-Set Selection for Effective Data Selection

This repository contains the source code of the Skoltech ML2021 course project "Core-Set Selection for Effective Data Selection".

### Structure and contents
* `src/`
  * `greedy_k_centers` contains the implementation of the Greedy K-Centers algorithm, with some utility functions.
  * `Glister` contains the implementation of all GLISTER algorithms, with some utility functions.
* `dataset/` contains a dataset manager. Since we are using CIFAR-10 throughout this repository, we unified access to the data set and wrote a data manager for maintaining access to the data set, so we don't have copies stored in every folder.
* `results/` contains the main results of our work, together with visualization code implemented in jupyter notebooks.
* `submodules/` contains other repositories we used, linked as submodules to this repo.

### Getting started
#### Dependencies

In order to use the repository, you need to have the following packages installed:
* `pytorch 1.8.0`
* `torchvision 0.9.0`

It is also highly recommended to use CUDA GPU cores for the training, as they are computationally very demanding.

#### Commands

### Documentation
The documentation can be found here:

https://www.overleaf.com/7232121456hvjtgyvnfwcd

### Project team
- Julia Tukmacheva
- Vladimir Omelyusik
- Roland Konlechner

### Relevant papers
Selection via Proxy: https://arxiv.org/abs/1906.11829

GLISTER: https://arxiv.org/abs/2012.10630
