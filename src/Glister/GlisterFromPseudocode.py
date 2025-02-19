## This script contain implementation of the Glister approach from pseudocode.
## Model definition was adapted from here: https://github.com/dssresearch/GLISTER/blob/master/models/simpleNN_net.py
## Created by Knowhere team, Skoltech 2021.

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

# This is the 2-linear-layered neural net used for small data sets, as proposed by the authors:
class ShallowNet(torch.nn.Module):
    def __init__(self, input_dim, num_class, theta=None):
        super(ShallowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 100)
        self.fc2 = torch.nn.Linear(100, num_class)

        if theta != None:
          self.fc2.weight.data = theta

    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.ReLU()(out)
        out = self.fc2(out)
        return out
    
# This class is used for convenient data extraction:
class Dat(torch_data.Dataset):
    def __init__(self, X, y):
        super(Dat, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):
        
        return self.X[idx], self.y[idx]


def compute_grad_train(observations, theta, input_dim, n_classes):
  '''
  Attributes:
  ---
  observations: torch.tensor
    One observation selected in V: observations[0]: features, observations[1]: target.
  theta: torch.tensor
    Initialization parameters.
  input_dim: int
    Input dimension size.
  n_classes: int
    Number of classes.
  ---
  Returns gradient of the loss by theta in theta and net.
  '''
  feats = observations[0].reshape(1, input_dim)
  target = observations[1].reshape(1,)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = ShallowNet(input_dim, n_classes, theta.reshape(n_classes, 100)) # add thetas as the weights on the last layer (called "last layer approximation" by authors)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

  net.train()
  optimizer.zero_grad()
  x = net(feats)
  train_loss = criterion(x, target)
  train_grad = torch.autograd.grad(train_loss, net.fc2.weight, retain_graph=True)[0]
  train_loss.backward()
  optimizer.step()

  return train_grad.reshape(1, n_classes*100), net

def compute_grad_val(net, observations, theta, input_dim, n_classes):
  '''
  Attributes:
  ---
  observations: torch.tensor
    One observation selected in V: observations[0]: features, observations[1]: target.
  theta: torch.tensor
    Initialization parameters.
  input_dim: int
    Input dimension size.
  n_classes: int
    Number of classes.
  ---
  Returns gradient of the loss by theta in theta and validation loss.
  '''
  val_loader = DataLoader(observations, batch_size=40, shuffle=False)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = ShallowNet(input_dim, n_classes, theta.reshape(n_classes, 100)) # add thetas as the weights on the last layer (called "last layer approximation" by authors)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
  # do the forward pass
  
  net.train()
  for X, target in val_loader:
    optimizer.zero_grad()
    x = net(X)
    val_loss = criterion(x, target)
    val_grad = torch.autograd.grad(val_loss, net.fc2.weight, retain_graph=True)[0]
    val_loss.backward()
    optimizer.step()

  return val_grad.reshape(1, n_classes*100), val_loss

def GreedyDSS(U, Val, theta_prev, eta, k, r, lambd, R, sel):
  '''
    Implementation of GreedyDSS (Algorithm 2) from GLISTER paper

    Attributes:
    ---
    U: torch.tensor
      Training data.
    Val: torch.tensor
      Validation data.
    theta_0: torch.tensor
      Model parameters initialization.
    eta: float
      Learning rate.
    k: int
      Number of point for which the model would be trained.
    r: int
      Number of Taylor approximations.
    lambd: float
      Regularization coefficient.
    R: function
      Regularization function.
    sel: str
      Selection method.

    Returns
    ---
    S: ndarray
      Coreset.
  '''
  #eps=500
  t = 0
  S = [U[np.random.randint(len(U))], U[np.random.randint(len(U))]]
  theta = theta_prev
  total_idxs = [*range(0, len(U))]

  while t < r:
    # V-data selection by two options
    if sel == "naive_greedy":
      V = Dat(U[total_idxs][0], U[total_idxs][1])
    elif sel == "stochastic_greedy":
      random_idxs = list(random.sample(total_idxs, round(len(U)*0.1))) # it seems that eps affects the time and accuracy of algorithm, and since it was not specified by authors, we set it to 10% of all training observations
      V = Dat(U[random_idxs][0], U[random_idxs][1])

    g_hats = np.array([])
    
    for e in V:
      grad_train, net = compute_grad_train(e, theta, input_dim, n_classes)
      theta_t_e = theta + eta *  grad_train
      grads_s = np.array([])

      for i in S:
        grad_s, _ = compute_grad_train(i, theta_t_e, input_dim, n_classes)
        grads_s = np.append(grads_s, grad_s)
      grads_s = np.array(grads_s).reshape(grads_s.shape[0]//(n_classes*100), n_classes*100)
      theta_s = theta + eta * torch.Tensor(np.sum(grads_s, axis=0))

      grad_val, val_loss = compute_grad_val(net, Val, theta_s, input_dim, n_classes)
      g_hats = np.append(g_hats, val_loss.detach().numpy() + eta * torch.matmul(grad_train, grad_val.T).detach().numpy()[0][0]) + lambd*R # g hats is np.array # the largest values, no regularization as set R, lambd=0
    
    g_hats = np.array(g_hats)
    
    best_indices = np.argpartition(np.array(g_hats), -round(k/r))[-round(k/r):] # finding k/r best indices.

    S.extend([V[i] for i in best_indices]) # add the corresponding indices observations to S.

    if sel == "stochastic_greedy":
      subset_idxs = list(np.array(random_idxs)[best_indices])

    else:
      subset_idxs = list(np.array(total_idxs)[best_indices])

    total_idxs = list(set(total_idxs).difference(set(subset_idxs))) # delete best indices from total indices.

    grads_theta = torch.zeros(n_classes*100).reshape(1, n_classes*100)

    for elem in [V[i] for i in best_indices]:
      grad, _ = compute_grad_train(elem, theta, input_dim, n_classes) # computing gradients for each new observations, corresponding to best indices

      grads_theta += grad

    theta = theta + grads_theta

    t += 1

  return S

def glister_online(U, Val, S_0, k, theta_prev, eta, T, L, r, lambd, R, sel):
  '''
  Attributes:
  ---
  U: torch.tensor
    Training data.
  Val: torch.tensor
    Validation data.
  S_0: torch.tensor
    Initial subset.
  k: int
    Size of the initial subset.
  theta_prev: torch.tensor
    Model parameter initialization.
  eta: float
    Learning rate.
  T: int
    Total epochs.
  L: int
    Epoch interval for selection.
  r: int
    Number of Taylor approximations.
  lambd: float
    Regularization coefficient.
  R: function
    Regularization function.
  sel: str
    Selection Method.
  
  Returns:
  ---
  S_T: torch.tensor
    Final subset
  theta_T: torch.tensor
    Parameters.
  '''
  theta = theta_prev
  S_t = S_0

  for t in range(T):
    if t % L == 0:
      print("Progress: {t}/{T}".format(t=t, T=T))
      S_t = GreedyDSS(U=U, Val=Val, theta_prev=theta, eta=eta, k=k, r=r, lambd=0, R=0, sel=sel)

    model = ShallowNet(input_dim, n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(S_t, batch_size=20, shuffle=True)
    
    # Performing one epoch of batch SGD
    model.train()
    for X, y in loader:
      
      optimizer.zero_grad()
      x = model(X)
      train_loss = criterion(x, y)
      train_loss.backward()
      optimizer.step()
      
    theta = model.fc2.weight.reshape(1, n_classes*100)

  return S_t, model.fc2.weight

def train(epochs, net, criterion, optimizer, train_loader, val_loader, verbose=True, device='cpu'):
    net.to(device)
    freq = max(epochs//15,1)
 
    for epoch in range(1, epochs+1):
        net.train()

        losses_train = []
        for X, target in train_loader:

            #X, target = X.to(device), target.to(device)
            
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
                #X, target = X.to(device), target.to(device)
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
            
            print('Val epoch {}'.format(epoch), ', Loss : {:.3}'.format(mean_train), ', Accuracy on test: {:.3}'.format(accuracy_score(y_true_val, y_pred_val)))

            
def experimental_setting(train_data, val_data, test_data, theta_prev, eta, k, r, lambd, R, sel, input_dim, n_classes, T, L, batch_size, network):
  print("Glister online:")
  start = time.perf_counter()
  # Select the subset
  subset = glister_online(U=train_data, Val=val_data, theta_prev=theta_prev, S_0 = train_data[[np.random.randint(len(train_data)), np.random.randint(len(train_data))]], eta=eta, k=k, r=r, lambd=lambd, R=R, sel=sel, T=T, L=L)
  print("time elapsed: ", time.perf_counter()-start)
  print("Testing the model:")
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = network
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

  train_loader = DataLoader(subset[0], batch_size=batch_size, shuffle=True) # load data with the selected subset
  val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False)

  train(200, net, criterion, optimizer, train_loader, test_loader, verbose=True)

  print("Random comparison:")

  random_sample = random.sample(list(train_data), k)
  random_samp_loader = DataLoader(random_sample, batch_size=batch_size, shuffle=True)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net_2 = ShallowNet(input_dim, n_classes)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net_2.parameters(), lr=0.05)
  train(200, net_2, criterion, optimizer, random_samp_loader, test_loader, verbose=True)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits, targets = load_digits(return_X_y=True)
digits = digits.astype(np.float32) / 255   # scaling

digits_train, digits_test, targets_train, targets_test = train_test_split(digits, targets, random_state=0)
digits_tr, digits_val, targets_tr, targets_val = train_test_split(digits_train, targets_train, random_state=0, shuffle=False)

input_dim = 8*8
n_classes = 10

train_digits = Dat(digits_tr, targets_tr)
val_digits = Dat(digits_val, targets_val)
test_digits = Dat(digits_test, targets_test)

# ---------- SKLEARN DIGITS data set ------------------------------------------------------------------------------------------------------------------------------------------------
n_classes = 10
input_dim = 64
#print("Approximately 10% of data")
#experimental_setting(train_digits, val_digits, test_digits, torch.randn(1, 1000), 0.05, 100, 50, 0, 0, "naive_greedy", 64, 10, 200, 20, 20, ShallowNet(64, 10)) # ~10% of data, no regularization

print("Approximately 30% of data")
experimental_setting(train_digits, val_digits, test_digits, torch.randn(1, 1000), 0.05, 300, 9, 0, 0, "naive_greedy", 64, 10, 200, 20, 20, ShallowNet(64, 10)) # ~30% of data, no regularization

print("Approximately 50% of data")
experimental_setting(train_digits, val_digits, test_digits, torch.randn(1, 1000), 0.05, 500, 15, 0, 0, "naive_greedy", 64, 10, 200, 20, 20, ShallowNet(64, 10)) # ~50% of data, no regularization



