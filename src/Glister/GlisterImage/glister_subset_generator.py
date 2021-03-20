import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms

from GlisterImage import GlisterOnlineImage

sys.path.append('../../dataset/')
import dataset_manager

class CifarNet(torch.nn.Module):
	def __init__(self):
		super(CifarNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(3,   64,  3)
		self.conv2 = torch.nn.Conv2d(64,  128, 3)
		self.conv3 = torch.nn.Conv2d(128, 256, 3)
		self.pool = torch.nn.MaxPool2d(2, 2)
		self.fc1 = torch.nn.Linear(64 * 4 * 4, 128)
		self.fc2 = torch.nn.Linear(128, 256)
		self.fc3 = torch.nn.Linear(256, 10)

	def forward(self, x):
		x = self.pool(torch.nn.functional.relu(self.conv1(x)))
		x = self.pool(torch.nn.functional.relu(self.conv2(x)))
		x = self.pool(torch.nn.functional.relu(self.conv3(x)))
		x = x.view(-1, 64 * 4 * 4)
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x


def generate_subset_indices(dataset_type='subset', frac_of_full_set=0.1):
	""" Generates subset indices from the full dataset with the GLISTER method.

	Keyword arguments:
	dataset_type -- Either 'fullset' to generate GLISTER indices of full dataset, or 'subset' to generate GLISTER indices of subset.
		Warning: 'fullset' is computationally very expensive.
	frac_of_full_set -- Fraction of the latent space datapoints the method generates
	
	"""

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	if(dataset_type == 'fullset'):
		trainset = dataset_manager.CIFAR10_full(dataset_manager.__file__, transform=transform)

	elif(dataset_type == 'subset'):
		trainset = dataset_manager.CIFAR10_Subset(dataset_manager.__file__, transform=transform)
	else:
		print("glister: generate_subset_indices: dataset type error: You have to choose either 'fullset' or 'subset'.")
		return

	testset = dataset_manager.CIFAR10_full(dataset_manager.__file__, train=False, transform=transform)

	fullset = trainset
	valset = trainset

	glister = GlisterOnlineImage(
	fullset = fullset,
	valset = valset,
	testset = testset,
	device = "cpu",
	validation_set_fraction = 0.1,
	trn_batch_size = 128,
	val_batch_size = 256,
	tst_batch_size = 256,
	dss_batch_size = 256,
	model = CifarNet(),
	num_epochs = 10,
	learning_rate = 0.05,
	num_classes = 10,
	n_channels = 3,
	bud = frac_of_full_set * len(trainset),
	lam = 1)

	val_acc, tst_acc, subtrn_acc, full_trn_acc,\
	val_loss, test_loss, subtrn_loss, full_trn_loss,\
	val_losses, substrn_losses, fulltrn_losses,\
	idxs, time = glister.random_greedy_train_model_online_taylor(np.arange(20))

	indices = np.array(idxs)

	np.savetxt('indices/glister_indices_{}.csv'.format(int(frac_of_full_set*100)), indices, delimiter=',')

