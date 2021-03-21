import sys
sys.path.append('../../../submodules/PyTorch_CIFAR10/')
sys.path.append('../../../dataset/')
sys.path.append('../../../submodules/PyTorch_CIFAR10/cifar10_models/')
sys.path.append('../../../dataset/')
sys.path.append('../../Glister/GlisterImage/indices/')
sys.path.append('../../greedy_k_centers/')

from datetime import datetime
from sklearn.metrics import accuracy_score
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from schduler import WarmupCosineLR
import numpy as np
import dataset_manager

from resnet import resnet18
from mobilenetv2 import mobilenet_v2
from densenet import densenet121
from vgg import vgg11_bn

def __train_model_epochs(epochs,
                         net,
                         net_name,
                         criterion,
                         optimizer,
                         train_loader,
                         val_loader,
                         percentage_of_dataset,
                         selector_name,
                         scheduler=None,
                         verbose=True,
                         device='cuda'):
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
    
    test_dataset = dataset_manager.CIFAR10_full(
        dataset_manager.__file__,
        train=False,
        transform=transform
    )

    return [train_dataset[i] for i in indices], test_dataset


def train_and_save_model(model_name = 'resnet',
                         coreset_selector = 'k-centers',
                         coreset_percentage = 0.1,
                         trainset_size = 'subset',
                         device = 'cuda'
                        ):
    """Trains every model in <models> input array and saves resulting weights. Uses other parameters to generate verbose console outputs during training.
    
    Keyword arguments:
    model_name -- name of model to be trained. Must be either 'resnet', 'mobilenet', 'vgg' or 'densenet'.
    coreset_selector -- name of coreset selector. Must be either 'glister', 'k-centers' or 'random'.
    coreset_percentage -- size of coreset as percentage of total train set. Must bei either '0.1', '0.3', '0.5' or '1.0'. If '1.0' is chosen, the model is trained on the full dataset, and coreset_selector is ignored.
    trainset_size -- choose wether you want to train on full CIFAR10 or a subset of CIFAR10 (20000 datapoints)
    
    """
    
    # Data
    
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
        
    if trainset_size == 'fullset':
        train_dataset = dataset_manager.CIFAR10_full(
            dataset_manager.__file__,
            train=True,
            transform=transform
        )
    elif trainset_size == 'subset':
        train_dataset = dataset_manager.CIFAR10_subset(
            dataset_manager.__file__,
            transform=transform
        )
    else:
        print("Dataset not supported, must be either 'fullset' or 'subset'.")
        return
    
    test_dataset = dataset_manager.CIFAR10_full(
        dataset_manager.__file__,
        train=False,
        transform=transform
    )
    
    # Models
    if model_name=='resnet':
        model = resnet18()
    elif model_name=='mobilenet':
        model = mobilenet_v2()
    elif model_name=='vgg':
        model = vgg11_bn()
    elif model_name=='densenet':
        model = densenet121()
    else:
        print("Model not supported. Must be either 'resnet', 'mobilenet', 'vgg' and 'densenet'.")
        return
        
        
    # Coreset selectors
    if coreset_percentage != 0.1 and coreset_percentage != 0.3 and coreset_percentage != 0.5 and coreset_percentage != 1.0:
        print("Coreset size not supported. Must be either 0.1, 0.3, 0.5 or 1.0")
        return
    
    if coreset_percentage != 1.0:
        if coreset_selector=='glister':
            coreset_str = str(int(coreset_percentage*100))
            train_indices = np.loadtxt('../../Glister/GlisterImage/indices/glister_indices_{}_{}.csv'.format(trainset_size, coreset_str), delimiter=',').astype(int)

        elif coreset_selector=='k-centers':
            train_indices = np.loadtxt('../../greedy_k_centers/k_centers_indices_{}.csv'.format(trainset_size), delimiter=',')[:int(len(train_dataset)*coreset_percentage)].astype(int)

        elif coreset_selector=='random':
            train_indices = np.arange(0, len(train_dataset))

            train_indices = np.random.choice(train_indices, size=int(len(train_dataset)*coreset_percentage), replace=False)
        else:
            print("Coreset selector not supported. Must be either 'glister', 'k-centers' or 'random'.")
            return
            
        train_dataset = [train_dataset[i] for i in train_indices]
        
    print("Size of selected dataset: ", len(train_dataset))
        
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
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
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("Using device: ", device)
    
    # Training parameters
    num_epochs = 100
    learning_rate = 1e-2
    weight_decay = 1e-2
    total_steps = num_epochs * len(train_loader)
    

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
    accuracies = __train_model_epochs(num_epochs, model, model_name, criterion, optimizer, train_loader, test_loader, str(int(coreset_percentage * 100)), coreset_selector, scheduler=scheduler, verbose=True, device=device)

    # take model as parameter, not cacluclate it

    if coreset_selector == None:
        weights_filename = 'model_weights/{:03}_{}_{}.pt'.format(int(coreset_percentage*100), model_name, trainset_size)
        results_filename = 'accuracy_results/{:03}_{}_{}.csv'.format(int(coreset_percentage*100), model_name, trainset_size)
    else:
        weights_filename = 'model_weights/{:03}_{}_{}_{}.pt'.format(int(coreset_percentage*100), model_name, coreset_selector, trainset_size)
        results_filename = 'accuracy_results/{:03}_{}_{}_{}.csv'.format(int(coreset_percentage*100), model_name, coreset_selector, trainset_size)
    torch.save(model.state_dict(), weights_filename)

    np.savetxt(results_filename, accuracies, delimiter=',')