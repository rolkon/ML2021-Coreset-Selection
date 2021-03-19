import sys
sys.path.append('../../submodules/PyTorch_CIFAR10/cifar10_models/')
sys.path.append('../../dataset/')
from resnet import resnet18
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import dataset_manager
from tqdm import tqdm

def generate_latent_space(dataset_type='fullset'):
    """ Generates the latent space of CIFAR10 using the last hidden layer of a trained ResNet.
    Keyword arguments:
    dataset_type -- Either 'fullset' to generate latent space on full dataset, or 'subset' to generate latent space on subset of 20000
    
    Outputs:
    Saves resulting latent space as CSV file.
    
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    #preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if(dataset_type == 'fullset'):
        train_dataset = dataset_manager.CIFAR10_full(dataset_manager.__file__, transform=transform)
        
    elif(dataset_type == 'subset'):
        train_dataset = dataset_manager.CIFAR10_Subset(dataset_manager.__file__, transform=transform)
    else:
        print("generate_latent_space: dataset type error: You have to choose either 'fullset' or 'subset'.")
        return

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    if device == 'cpu':
        print("Warning: We did not detect a CUDA core. This could take a while.")

    net = resnet18(pretrained=True)

    #transform all datapoints to a latent representation
    latent_data = []

    print("Generating latent space...")
    with tqdm(total=len(train_dataset)) as pbar:
        for i in range(len(train_dataset)):
            datapoint = train_dataset[i][0].reshape(1, 3, 32, 32)
            latent_data.append(net.encode(datapoint).detach().numpy()[0])
            pbar.update(1)

    filename = 'CIFAR10_latent_data_{}.csv'.format(dataset_type)
    print("Saving ", filename, "...")
    latent_data = np.array(latent_data)
    np.savetxt(filename, latent_data, delimeter=',')