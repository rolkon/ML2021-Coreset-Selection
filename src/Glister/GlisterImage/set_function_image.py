## This script contain implementation of SetFunctionImage class, which is a version of GreedyDSS algorithm.
## This class is adopted from https://github.com/dssresearch/GLISTER/blob/master/models/set_function_grad_computation_taylor.py
## The changes concern the handling of data shapes.
## If you encounter problems with shapes, the problem may be in our code, otherwise, 
## look at the original repository.
## Adapted by Knowhere team, Skoltech 2021.

# Imports
import numpy as np
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, SequentialSampler, BatchSampler
from queue import PriorityQueue
from torch import random

class SetFunctionImage(object):

    def __init__(self, trainset, x_val, y_val, model, loss_criterion,
                 loss_nored, eta, device, num_channels, num_classes, batch_size):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = loss_criterion  # Make sure it has reduction='none' instead of default
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size

    # Calculte per-element gradients
    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array([list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        cnt = 0

        for batch_idx in batch_wise_indices:
            inputs = torch.cat([self.trainset[x][0].reshape(-1, self.num_channels, self.trainset[x][0].shape[1], self.trainset[x][0].shape[2]) for x in batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)

            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1,1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat((data, F.softmax(self.model(inputs), dim=1)), dim=0)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)

        grads_vec = data - outputs
        torch.cuda.empty_cache()
        self.grads_per_elem = grads_vec

    # Update validation gradients
    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()

        if first_init:
            with torch.no_grad():
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(self.device)
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(grads_val[0] * (params[-1].data - self.eta * grads_elem[0]))
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Main function, implementation of DSS
    def naive_greedy_max(self, budget, theta_init):

        # Compute per-elements gradients
        self._compute_per_element_grads(theta_init)

        # Init validation set gradients
        self._update_grads_val(theta_init, first_init=True)

        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))

        while (numSelected < budget):
            t_one_elem = time.time()

            # Select subset
            subset_selected = list(np.random.choice(np.array(list(remainSet)), size=subset_size, replace=False))

            # Calculate gradients
            rem_grads = [self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0]) for x in subset_selected]

            # Calculate gains
            gains = self.eval_taylor_modular(rem_grads, theta_init)

            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)

            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[bestId]  # Making it a list so that is mutable!

            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)

            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print("numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem)
            numSelected += 1
        
        return list(greedySet), grads_currX
