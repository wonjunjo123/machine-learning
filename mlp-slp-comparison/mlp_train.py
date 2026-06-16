

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from mlp import BaseClassifier
import time

# For reproducability
torch.manual_seed(0)

train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


in_dim, feature_dim, out_dim = 784, 256, 10


def train(classifier,optimizer,epochs,loss_fn):
    classifier.train()
    loss_lt = []
    tolerance = 0.26
    epoch = 0
    
    for epoch in range(epochs):
    #while True:
        running_loss = 0.0
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Keep track of sum of loss of each minibatch
            running_loss += computed_loss.item()
        train_loss = running_loss/len(train_loader)
        loss_lt.append(train_loss)
        print("Epoch: {} train loss: {}".format(epoch+1, train_loss))
        #epoch += 1
        #if train_loss < tolerance: break

    '''plt.plot([i for i in range(1,epochs+1)], loss_lt)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MNIST Training Loss: optimizer {}, lr {}, epochs {}".format("SGD", lr, epochs))
    plt.show()'''
    # Save state to file as checkpoint
    print("Saving network in mnist_mlp.pt") 
    torch.save(classifier.state_dict(), 'mnist_mlp.pt')



if __name__ == "__main__":

    # Instantiate model, optimizer, and hyperparameter(s)
    lr=1e-2 # used to be 1e-3, changed to 1e-2 for assignment 5
    loss_fn = nn.CrossEntropyLoss()
    epochs=500 # used to be 40, changed to 500 for assignment 5
    classifier = BaseClassifier(in_dim, feature_dim, out_dim)
    optimizer = optim.SGD(classifier.parameters(), lr=lr)

    # just to keep track of time (Used claude to figure out which command to use)
    # https://claude.ai/chat/aadf0d4f-f90d-4c5a-a7c2-6d09a9c0edc6
    start = time.perf_counter()

    # Actual training
    train(classifier,optimizer,epochs,loss_fn)
    
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.3f} seconds")


