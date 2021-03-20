import sys
sys.path.append('../../../submodules/PyTorch_CIFAR10/')
sys.path.append('../../../dataset/')

from datetime import datetime
from sklearn.metrics import accuracy_score
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from schduler import WarmupCosineLR
import numpy as np
import dataset_manager

def train_model_epochs(epochs, net, net_name, criterion, optimizer, train_loader, val_loader, percentage_of_dataset, selector_name, scheduler=None, verbose=True, device='cuda'):
    """Trains model for <epochs> number of epochs, returns accuracies after last epoch. Very verbose."""

    freq = max(epochs//20,1)
    
    accuracies = []
 
    for epoch in range(1, epochs+1):
        net.train()

        losses_train = []
        for X, target in train_loader:
            X, target = X.to(device), target.to(device)
            
            net_output = net(X)
            loss = criterion(net_output, target)
            losses_train.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #scheduler is cosine annealing, so its called in the step loop
            if scheduler != None:
                scheduler.step()
        
        if verbose and epoch%freq==0:

            y_pred_val =  []
            y_true_val = []
            net.eval()
            losses_val = []
            
            for X, target in val_loader:
                X, target = X.to(device), target.to(device)
                
                # Compute the validation loss
                target_hat_val = net(X)
                loss = criterion(target_hat_val, target)
                losses_val.append(float(loss))
                
                y_pred_val.extend(target_hat_val.argmax(1).tolist())
                y_true_val.extend(target.tolist())

            mean_val = sum(losses_val)/len(losses_val)
            mean_train = sum(losses_train)/len(losses_train)
            
            accuracies.append(accuracy_score(y_true_val, y_pred_val))
            
            print('Timestamp: ', datetime.now().strftime("%H:%M:%S"), \
                '\tVal epoch {}'.format(epoch), \
                '\n\tModel: {}'.format(net_name), \
                '\n\tSelector:{}'.format(selector_name), \
                '\n\tPercentage of dataset:{}'.format(percentage_of_dataset), \
                '\n\tLoss Train: {:.3}'.format(mean_train), \
                ',\n\tLoss Test: {:.3}'.format(mean_val),\
                ',\n\tAccuracy on test: {:.3}'.format(accuracy_score(y_true_val, y_pred_val)) )
            
    return accuracies

def __get_datasubset(indices):
    #factors selected from torch docs
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    #preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
        
    train_dataset = dataset_manager.CIFAR10_full(
        dataset_manager.__file__,
        train=True,
        transform=transform
    )

    return train_dataset[indices]


def train_and_save_models(models, model_names, train_indices, percentage_of_dataset, selector_name=None):
    """Trains every model in <models> input array and saves resulting weights. Uses other parameters to generate verbose console outputs during training."""
    
    train_datasubset = __get_datasubset(train_indices)
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_datasubset,
                                               batch_size=128, 
                                               shuffle=True,
                                               drop_last=True
                                              )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128, 
                                              shuffle=False,
                                              drop_last=True
                                             )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training parameters
    num_epochs = 100
    learning_rate = 1e-2
    weight_decay = 1e-2
    total_steps = num_epochs * len(train_loader)
    
    #train selected models on subset
    for model, label in zip(models, model_names):

        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True
        )
        
        # Scheduler
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_epochs=total_steps * 0.3,
            max_epochs = total_steps)

        # Train the model
        accuracies = train_model(num_epochs, model, label, criterion, optimizer, train_loader, test_loader, percentage_of_dataset, selector_name, scheduler=scheduler, verbose=True, device=device)
        
        # take model as parameter, not cacluclate it
        # percentage_of_dataset = np.round((len(train_datasubset)/len(train_dataset))*100).astype(int)

        if selector_name == None:
            weights_filename = 'model_weights/{:03}_{}.pt'.format(percentage_of_dataset, label)
            results_filename = 'accuracy_results/{:03}_{}.csv'.format(percentage_of_dataset, label)
        else:
            weights_filename = 'model_weights/{:03}_{}_{}.pt'.format(percentage_of_dataset, label, selector_name)
            results_filename = 'accuracy_results/{:03}_{}_{}.csv'.format(percentage_of_dataset, label, selector_name)
        torch.save(model.state_dict(), weights_filename)
        
        np.savetxt(results_filename, accuracies, delimiter=',')