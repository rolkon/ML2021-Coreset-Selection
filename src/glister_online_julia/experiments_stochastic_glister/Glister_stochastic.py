# General imports
import copy, datetime, os, subprocess, sys, time, math
import numpy as np

# Everything torch-related
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
import torch.utils.data as torch_data
import torchvision
import torch.nn.functional as F

# Custom classes
from set_function import SetFunction
from mnist_net import MnistNet

import random

class GlisterOnline():
    
    def __init__(self, fullset = None, valset = None, testset = None, 
                 device = 'cpu', validation_set_fraction = 0.1,
                 trn_batch_size = 20, val_batch_size = 20, tst_batch_size = 20, dss_batch_size = 20,
                 model = None, num_epochs = 200, learning_rate = 1.0, num_classes = None, n_channels = 1,
                 bud = 100, lam = 0.1, r=100):
        '''
        Attributes:
        ------
        fullset: torch_data.Dataset
            Training set.
            
        valset: torch_data.Dataset
            Validation set (= Training set).
            
        testset: torch_data.Dataset
            Test set.
            
        device: str
            Device to use.
        
        validation_set_fraction: float
            Fraction of validation set wrt train set.
            
        trn_batch_size, val_batch_size, tst_batch_size, dss_batch_size: int
            Sizes of train, validation, test, dss batch sizes.
        
        model: nn.Module
            Model to use.
        
        bud: int
            Budget.
        
        lam: float
            Lambda.
        '''
        
        # Set "global" variable
        self.fullset = fullset
        self.valset = valset
        self.testset = testset
        self.device = device
        self.validation_set_fraction = validation_set_fraction
        self.trn_batch_size = trn_batch_size
        self.val_batch_size = val_batch_size
        self.tst_batch_size = tst_batch_size
        self.dss_batch_size = dss_batch_size
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.bud = bud
        self.lam = lam
        self.r = r
        
        # Separate train, validation, test sets
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, validset = random_split(fullset, [num_trn, num_val])
        self.trainset = trainset
        self.validset = validset
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = trn_batch_size,
                                                  shuffle = False, pin_memory = False)

        valloader = torch.utils.data.DataLoader(valset, batch_size = val_batch_size, shuffle = False,
                                                sampler = SubsetRandomSampler(validset.indices),
                                                pin_memory = False)

        testloader = torch.utils.data.DataLoader(testset, batch_size = tst_batch_size, 
                                                 shuffle = False, pin_memory = False)

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        
        # Transform to tensors
        trainset_idxs = np.array(trainset.indices)
        batch_wise_indices = list(BatchSampler(SequentialSampler(trainset_idxs), 1000, drop_last=False))
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat([fullset[x][0].reshape(1, -1) for x in batch_idx], dim=0).type(torch.float)
            targets = torch.tensor([fullset[x][1] for x in batch_idx])
            if cnt == 0:
                x_trn = inputs
                y_trn = targets
                cnt = cnt + 1
            else:
                x_trn = torch.cat([x_trn, inputs], dim=0)
                y_trn = torch.cat([y_trn, targets], dim=0)
                cnt = cnt + 1

        for batch_idx, (inputs, targets) in enumerate(valloader):
            if batch_idx == 0:
                x_val = inputs
                y_val = targets
                x_val_new = inputs.reshape(val_batch_size, -1)
            else:
                x_val = torch.cat([x_val, inputs], dim=0)
                y_val = torch.cat([y_val, targets], dim=0)
                x_val_new = torch.cat([x_val_new, inputs], dim=0)
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx == 0:
                x_tst = inputs
                y_tst = targets
                x_tst_new = inputs.view(tst_batch_size, -1)
            else:
                x_tst = torch.cat([x_tst, inputs], dim=0)
                y_tst = torch.cat([y_tst, targets], dim=0)
                x_tst_new = torch.cat([x_tst_new, inputs], dim=0)
                
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.x_val = x_val
        self.y_val = y_val
        self.x_tst = x_tst
        self.y_tst = y_tst
                
    def random_greedy_train_model_online_taylor(self, start_rand_idxs):
        '''
        Start GlisterDSS + GlisterOnline.
        
        Attributes:
        ------
        
        start_rand_idxs: ndarray
            Initial indexes.
            
        Returns:
        ------
        val_acc, 
        tst_acc,  
        subtrn_acc, 
        full_trn_acc, 
        val_loss, 
        test_loss, 
        subtrn_loss, 
        full_trn_loss, 
        val_losses, 
        substrn_losses, 
        fulltrn_losses, 
        idxs, 
        time

        '''
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize NN model
        model = self.model      
        model = model.to(self.device)
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs
        criterion = nn.CrossEntropyLoss()
        criterion_nored = nn.CrossEntropyLoss(reduction = 'none')
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
        
        # Initialize indexes
        idxs = start_rand_idxs
    
        total_idxs = list(np.arange(len(self.y_trn)))
        random_idxs = list(random.sample(total_idxs, round(len(self.y_trn)*0.1)))
        
        # Set GreedyDSS model
        setf_model = SetFunction(self.trainset, self.x_val, self.y_val, model, criterion,
                                 criterion_nored, learning_rate, self.device, self.n_channels, 
                                 self.num_classes, self.dss_batch_size)
        
        #print("Starting Randomized Greedy Online OneStep Run with taylor!")
        substrn_losses = np.zeros(self.num_epochs)
        fulltrn_losses = np.zeros(self.num_epochs)
        val_losses = np.zeros(self.num_epochs)
        subset_trnloader = torch.utils.data.DataLoader(self.trainset, batch_size = self.trn_batch_size, 
                                                       shuffle = False,
                                                       sampler = SubsetRandomSampler(idxs),
                                                       pin_memory = False)
        for i in range(self.num_epochs):
            actual_idxs = np.array(self.trainset.indices)[idxs]
            batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), 
                                                                            self.trn_batch_size, 
                                                                            drop_last = True))]
            subtrn_loss = 0
            for batch_idx in batch_wise_indices:
                inputs = torch.cat([self.fullset[x][0].reshape(1, -1) for x in batch_idx], dim=0).type(torch.float)
                targets = torch.tensor([self.fullset[x][1] for x in batch_idx])
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                subtrn_loss += loss.item()
                loss.backward()
                optimizer.step()

            val_loss = 0
            full_trn_loss = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.valloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    full_trn_loss += loss.item()

            substrn_losses[i] = subtrn_loss
            fulltrn_losses[i] = full_trn_loss
            val_losses[i] = val_loss
            #print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            #print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))

            subset_idxs, grads_idxs = setf_model.naive_greedy_max(budget = self.bud, theta_init = clone_dict)
            rem_idxs = list(set(random_idxs).difference(set(subset_idxs)))
            subset_idxs.extend(list(np.random.choice(rem_idxs, size=self.bud, replace=True)))
            idxs = subset_idxs
            #print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(self.trainset, batch_size = self.trn_batch_size, shuffle=False,
                                                               sampler = SubsetRandomSampler(idxs), num_workers=0,
                                                               pin_memory=True)

        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                subtrn_loss += loss.item()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
        subtrn_acc = subtrn_correct / subtrn_total

        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        val_acc = val_correct / val_total

        full_trn_loss = 0
        full_trn_correct = 0
        full_trn_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()
                _, predicted = outputs.max(1)
                full_trn_total += targets.size(0)
                full_trn_correct += predicted.eq(targets).sum().item()
        full_trn_acc = full_trn_correct / full_trn_total
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        tst_acc = correct / total

        print("SelectionRun---------------------------------")
        print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
        print("Validation Loss and Accuracy:", val_loss, val_acc)
        print("Test Data Loss and Accuracy:", test_loss, tst_acc)
        print('-----------------------------------')
        return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


