## This script contains an auxilary function for training a neural net model.
## Copied from HW2 of ML2021 Skoltech course.

import pandas as pd
import numpy as np
import torch
import torch.utils.data as torch_data
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import time

import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

def train(epochs, net, criterion, optimizer, train_loader, val_loader, verbose=True, device='cpu'):
    net.to(device)
    freq = max(epochs//15,1)
 
    for epoch in range(1, epochs+1):
        net.train()

        losses_train = []
        for X, target in train_loader:
            
            optimizer.zero_grad()
            x = net(X)
          
            train_loss = criterion(x, target)
            train_loss.backward()
            optimizer.step()
            losses_train.append(train_loss.item())

        if verbose and epoch%freq==0:
            y_pred_val =  []
            y_true_val = []
            net.eval()
            for X, target in val_loader:
                losses_val = []  

                optimizer.zero_grad()
                x = net(X)
                target_hat_val = torch.nn.Softmax(1)(x)

                val_loss = criterion(x, target)

                losses_val.append(val_loss.item())
                                
                y_pred_val.extend(target_hat_val.argmax(1).tolist())
                y_true_val.extend(target.tolist())

            mean_val = sum(losses_val)/len(losses_val)
            mean_train = sum(losses_train)/len(losses_train)

        if epoch == 200:
          print('Test Data Loss {}'.format(epoch), ', Loss : {:.3}'.format(mean_train), ', and Accuracy: {:.3}'.format(accuracy_score(y_true_val, y_pred_val)))
        else:
          pass