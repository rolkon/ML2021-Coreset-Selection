# Core-Set Selection for Effective Data Selection

This repository contains the source code of the Skoltech ML2021 course project "Core-Set Selection for Effective Data Selection".

## Abstract

Core-set selection techniques aim to select a subset of a given training data set, with the goal of maximizing the subsets utility in training a neural network. In this repository, we analyze the core-set selection methods **GLISTER-ONLINE** and **Greedy K-Centers** and compare them to a random core-set selection as the baseline quality measure. We replicated the results of previous papers and studied how well the produced core-sets generalize across different deep neural network architectures. In our findings, the tested core-set algorithms produced core-sets that were well generalizable across multiple neural networks. Furthermore, no significant accuracy gains could be achieved by using **GLISTER-ONLINE** or **Greedy K-Centers** over random core-set sampling.

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
With the parameter, either 'fullset' or 'subset' can be selected. This creates the latent space either from the full CIFAR-10 dataset (50'000 samples), or from the subset (20'000 samples). The result is saved in a CSV file in the same directory.

The resulting file will be several hundets of Megabytes in size, that's why it is not included in this repository. It can be might be added upon request.

### Creating core-set indices
#### Greedy K-Centers
To do this, make sure you have the latent space created.

Go to `src/greedy_k_centers` and execute the following:
```python
>>> from k_centers_coreset_generator import generate_coreset_indices
>>> generate_coreset_indices(dataset_type='fullset', frac_of_full_set=0.5)
```
As with the latent space creation, you can either create the core-set for the full CIFAR-10 data set (50'000), or for the data subset (20'000).

The `frac_of_full_set` parameter describes the size of the core-set with regards to the full dataset (or subset).

#### GLISTER

Go to `src/Glister/GlisterImage` and execute the following:
```python
>>> from glister_coreset_generator import generate_coreset_indices
>>> generate_coreset_indices(dataset_type='subset', frac_of_full_set=0.1)
```
Again, you can choose between `fullset` and `subset`. In our experience, we only used `subset`, as generating the GLISTER indices on the full set was unfeasible. Also, in GLISTER, `frac_of_full_set` only allows 3 settings: `0.1`, `0.3` and `0.5`.

For both Greedy K-Centers and GLISTER, the resulting core-sets will be saved as CSV files directly in the same directory.

### Training models
If you have generated the core-set indices of either Greedy K-Center, GLISTER, or both, you can train models and check the achieved accuracy.

For this, the models from the `submodules/PyTorch_CIFAR10` submodule will be used.

Go to `src/experiments/models_generalization` and execute the following:
```python
>>> from training_utilities import train_and_save_model
>>> train_and_save_model(model_name='resnet', coreset_selector='k-centers', coreset_percentage='0.1', trainset_size='subset', device='cuda')
```
* `model_name` lets you choose between `resnet`, `mobilenet`, `vgg` and `densenet`
* `coreset_selector` accepts `glister`, `k-centers` and `random`
* `coreset_percentage` accepts `0.1`, `0.3` and `0.5`
* `trainset_size` can be `fullset` or `subset`, as above
* `device` lets you select the device for the training. We would strongly recommend only executing this command if you have a CUDA core available.

This function trains the model in a very verbose way. The resulting accuracies over the 100 epochs are logged in `accuracy_results/`.

### Working with Jupyter Notebooks
Some jupyter notebooks are sprinkled across this repository, as we like to work with them for visualization and interactive purposes. If you want to use them, we'd suggest launching the notebook in the root of the repository, as to not break any path dependencies that might be there.

## Report
The report for this project can be found here:

https://www.overleaf.com/7232121456hvjtgyvnfwcd

## Project team
- Julia Tukmacheva
- Vladimir Omelyusik
- Roland Konlechner

## Relevant papers
* Selection via Proxy: https://arxiv.org/abs/1906.11829
* GLISTER: https://arxiv.org/abs/2012.10630
