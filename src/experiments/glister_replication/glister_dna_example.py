#!/usr/bin/env python3

## This script provides an example of using Glister on dna subset.
## Written by Knowhere team, 20.03.2021

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader

from train import train

###### Example with DNA ######
from GlisterRegular.GlisterNaive import GlisterNaive
from GlisterRegular.GlisterStochastic import GlisterStochasticReg, GlisterStochasticNoReg

torch.manual_seed(42)
np.random.seed(42)

# Prepare data
class DNA_DATA(torch_data.Dataset):
    
    def __init__(self, X, y):
        super(DNA_DATA, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) 
    
    def __len__(self):
        return list(self.X.size())[0]
    
    def __getitem__(self, idx):
        return (self.X[idx, :], self.y[idx])

data = pd.read_csv('dna.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis = 1), data['class'], test_size = 0.3)
y_train = y_train - 1
y_test = y_test - 1

fullset = DNA_DATA(np.array(X_train), np.array(y_train))
valset = DNA_DATA(np.array(X_train), np.array(y_train))
testset = DNA_DATA(np.array(X_test), np.array(y_test))

# Set up the model
class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(180, 20)
        self.linear2 = torch.nn.Linear(20, 3)
    
    def forward(self, x, last=False):
        l1scores = torch.nn.functional.relu(self.linear1(x))
        scores = self.linear2(l1scores)
        if last:
            return scores, l1scores
        else:
            return scores

# Run Glister
glister_naive_acc = []
glister_naive_indexes = []

for k in [0.1, 0.3, 0.5]:
    glister_naive = GlisterNaive(
    fullset = fullset,
    valset = valset,
    testset = testset,
    device = "cpu",
    validation_set_fraction = 0.1,
    trn_batch_size = 20,
    val_batch_size = 50,
    tst_batch_size = 50,
    dss_batch_size = 50,
    model = TwoLayerNet(),
    num_epochs = 5,
    learning_rate = 0.05,
    num_classes = 3,
    n_channels = 1,
    bud = int(k * len(fullset)),
    lam = 0.1)
    
    val_acc, tst_acc, subtrn_acc, full_trn_acc,\
    val_loss, test_loss, subtrn_loss, full_trn_loss,\
    val_losses, substrn_losses, fulltrn_losses,\
    idxs, time = glister_naive.random_greedy_train_model_online_taylor(np.arange(20))
    
    glister_naive_acc.append(tst_acc)
    glister_naive_indexes.append(idxs)

print('Test accuracies for k = [0.1, 0.3, 0.5]: ', glister_naive_acc)
print('Selected indexes for k = [0.1, 0.3, 0.5]: ', glister_naive_indexes)

