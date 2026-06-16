
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn

from slp import BaseClassifier
from slp_train import in_dim, out_dim, test_dataset, test_loader



def test(classifier, loss_fn):
    classifier.eval()
    accuracy = 0.0
    computed_loss = 0.0

    confusion_matrix = torch.zeros((10,10))

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)

            # Get loss and accuracy
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)
            
            for tgt, pr in zip(target, preds):
                confusion_matrix[tgt.item(), pr.item()] += 1
                
        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))
            

    # In previous assignment, I used Claude to do print formatting
    # https://claude.ai/chat/5ba262a6-b532-4572-bf25-c97fc582fd32
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
    
    classifier = BaseClassifier(in_dim, out_dim)    
    classifier.load_state_dict(torch.load('mnist_slp.pt', weights_only = True))
    classifier.eval()

    test(classifier, loss_fn)
    
