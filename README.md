# Core-Set Selection for Effective Data Selection

This repository contains the source code of the Skoltech ML2021 course project "Core-Set Selection for Effective Data Selection".

## Structure and contents
* `src/`
  * `greedy_k_centers` contains the implementation of the Greedy K-Centers algorithm, with some utility functions.
  * `Glister` contains the implementation of all GLISTER algorithms, with some utility functions.
* `dataset/` contains a dataset manager. Since we are using CIFAR-10 throughout this repository, we unified access to the data set and wrote a data manager for maintaining access to the data set, so we don't have copies stored in every folder.
* `results/` contains the main results of our work, together with visualization code implemented in jupyter notebooks.
* `submodules/` contains other repositories we used, linked as submodules to this repo.

## Getting started
### Dependencies

In order to use the repository, you need to have the following packages installed:
* `pytorch 1.8.0`
* `torchvision 0.9.0`

It is also highly recommended to use CUDA GPU cores for most of the tasks described below, as they are computationally very demanding.

### Creating the latent space from CIFAR-10 for Greedy K-Centers

To do this, make sure the model weights in `submodules/PyTorch_CIFAR10` are downloaded. You can go there and execute the `download_weights.py` script to download them.

Go to `src/greedy_k_centers` and execute the following:

```python
>>> from latent_space_generator import generate_latent_space
>>> generate_latent_space(dataset_type='fullset')
```
With the parameter, either 'fullset' or 'subset' can be selected. This creates the latent space either from the full CIFAR-10 dataset (50'000 samples), or from the subset (20'000 samples). The result is saved in a CSV file.

### Creating core-set indices
#### Greedy K-Centers
To do this, make sure you have the latent space created.

Go to `src/greedy_k_centers` and execute the following:
```python
>>> from k_centers_subset_generator import generate_subset_indices
>>> generate_subset_indices(dataset_type='fullset', frac_of_full_set=0.5)
```
As with the latent space creation, you can either create the core-set for the full CIFAR-10 data set (50'000), or for the data subset (20'000).

The `frac_of_full_set` parameter describes the size of the core-set with regards to the full dataset (or subset).

#### GLISTER

### Training models

## Documentation
The documentation can be found here:

https://www.overleaf.com/7232121456hvjtgyvnfwcd

## Project team
- Julia Tukmacheva
- Vladimir Omelyusik
- Roland Konlechner

## Relevant papers
* Selection via Proxy: https://arxiv.org/abs/1906.11829
* GLISTER: https://arxiv.org/abs/2012.10630
