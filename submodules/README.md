# Submodules

For our coreset-selection work, we made use of Huy Phans' work. In his repository, he provides common deep learning
models fitted for and pre-trained on the CIFAR-10 dataset.

We forked from this repository made some changes:
* We added an `encode()` function to the Resnet18, so we are able to generate a latent space for Greedy K-Centers.
* We added a `download_weights.py` file to directly download the provided weights.
