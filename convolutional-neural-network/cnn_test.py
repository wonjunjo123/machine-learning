
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn

from cnn import MNISTConvNet
from cnn_train import testdataset, testloader



def test(model, loss_fn):
    model.eval()
    accuracy = 0.0
    computed_loss = 0.0

    confusion_matrix = torch.zeros((10,10))

    with torch.no_grad():
        for data, target in testloader:
            #data = data.flatten(start_dim=1)
            out = model(data)
            _, preds = out.max(dim=1)
            

            # Get loss and accuracy
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)
            
            for tgt, pr in zip(target, preds):
                confusion_matrix[tgt.item(), pr.item()] += 1
                
        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item()/(len(testloader)*64), accuracy*100.0/(len(testloader)*64)))
            

    # 
    cols = list(range(10))
    col_width = 6
    
    # Header
    print(f"{'Actual Digit':>15} |", end="")
    print(f"{'Classification Output':^65}")

    print(f"{'':>15} |", end="")
    for c in cols:
        print(f"{c:>{col_width}}", end="")
    print()

    print("-" * 16 + "+" + "-" * (col_width * 10))

    # Rows
    for i, row in enumerate(confusion_matrix):
        print(f"{i:>15} |", end="")
        for val in row:
            print(f"{int(val):>{col_width}}", end="")
        print()



if __name__ == "__main__":
    

    loss_fn = nn.CrossEntropyLoss()
    
    model = MNISTConvNet()    
    model.load_state_dict(torch.load('mnist_40_gpu.pt', weights_only = True))
    model.eval()

    test(model, loss_fn)
    
