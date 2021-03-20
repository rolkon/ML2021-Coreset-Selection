from torchvision.datasets import CIFAR10
import numpy as np
import os


# returns full CIFAR10-dataset
def CIFAR10_full(path ,train=True, transform=None, target_transform=None):
    """Returns the torch CIFAR10 dataset.
    
    Keyword arguments:
    path -- the relative path from your file to the module. See example below.
    train, transform, target_transform -- torch dataset options
    
    For the path, it's easiest if you use module.__file__, as can be seen in this example:
    
    >>> import dataset_manager
    >>> dataset = dataset_manager.CIFAR10_Subset(dataset_manager.__file__)
    
    """
    directory = os.path.dirname(path)
    dataset = CIFAR10(root=directory+'/data', train=train, transform=transform, target_transform=target_transform, download=True)
    
    return dataset

def CIFAR10_subset(path, size=20000, transform=None, target_transform=None):
    """Returns subset of <size> uniformly distributed samples from the torch CIFAR10 training set.
    
    Keyword arguments:
    path -- the relative path from your file to the module. See example below.
    size -- size of the subset. 20000 by default.
    transform, target_transform -- torch dataset options
    
    For the path, it's easiest if you use module.__file__, as can be seen in this example:
    
    >>> import dataset_manager
    >>> dataset = dataset_manager.CIFAR10_Subset(dataset_manager.__file__)
    
    """
    dataset = CIFAR10_full(path, train=True, transform=transform, target_transform=target_transform)
    
    #select uniform, seeded random sample of length <size> from the full dataset
    np.random.seed(0)
    train_indices = np.random.permutation(np.arange(0, len(dataset)))[:size]
    
    return [dataset[i] for i in train_indices]

def CIFAR10_subset_indices(size=20000):
    """Returns indices of subset of <size> uniformally distributed samples from the torch CIFAR10 training set.

    Keyword arguments:
    size -- size of the subset. 20000 by default.

    """
    np.random.seed(0)
    indices = np.random.permutation(np.arange(0, 50000))[:size]

    return indices