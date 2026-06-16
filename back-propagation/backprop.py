
'''
Build backpropagation model and use it for XOR, 2-not-2, and digit classification
Adapted from
  https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


Wonjun Jo
CSCI315
Professor Simon Levy
2/21/2026
'''

import numpy as np
from tqdm import tqdm
import pickle
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Model:
    
    def __init__(self, n, m, h, iters = 1000, eta = 0.5): # we are passing in rows and columns

        self.wih = np.random.randn(n+1, h)/2
        self.who = np.random.randn(h+1, m)/2
        self._h = h
        self._eta = eta
        self._iters = iters # don't actually use this because this is only for printing purposes, has no effect
        # actual iterations are controlled in train method
        
        
    def __str__(self):
        
        return 'Backprop model with {} inputs and {} outputs'.format(self.w.shape[0]-1, self.w.shape[1])

    def _forward(self, Inputs, j): # the _ in the front is to indicate a private method

        Ij = np.append(Inputs[j], [1])
        
        Hnetj = Ij @ self.wih
        Hj = sig(Hnetj)

        Onetj = (np.append(Hj, [1])) @ self.who
        Oj = sig(Onetj)
        
        return Ij, Hnetj, Hj, Onetj, Oj

    def train(self, Inputs, Targets, iters = 1000, eta=0.5):
        
        p = len(Inputs) # number of patterns
        for i in tqdm(range(iters)):

            dwih = np.zeros(self.wih.shape) # weight changes
            dwho = np.zeros(self.who.shape) # weight changes

            for j in range(p): # loop over patterns

                # "forward pass"
                Ij, Hnetj, Hj, Onetj, Oj = self._forward(Inputs, j)

                dOj = (Targets[j] - Oj) * dsig(Onetj)
                dHj = np.delete(dOj @ self.who.T, -1) * dsig(Hnetj) # Backprop
                
                dwih = dwih + np.outer(Ij, dHj) #learning rule
                dwho = dwho + np.outer(np.append(Hj, [1]), dOj)
        
            self.wih = self.wih + eta*dwih/p
            self.who = self.who + eta*dwho/p
            
                
    def test(self, Inputs, n=1):

        p = len(Inputs)
        results = np.zeros((p,n))
        
        for j in range(p):
            _, _, _, _, Oj = self._forward(Inputs, j)
            results[j] = Oj

        return results

    def save(self, fileName):

        fileObj = open(fileName, 'wb')
        pickle.dump((self.wih, self.who), fileObj)
        fileObj.close()
        
    def load(self, fileName): # loads a pickle file and makes its content to be its weights

        fileObj = open(fileName, 'rb')
        self.wih, self.who = pickle.load(fileObj)
        fileObj.close()
    

def sig(x):
    return 1/(1 + np.exp(-x))

def dsig(x):
    return sig(x)*(1-sig(x))

def show_pattern(pattern):
    for row in range(14):
        for col in range(14):
            print("*" if pattern[14*row+col] > 0 else ' ', end = ' ')
        print()

        
# format the file from txt to 2d array of input data and targets
def process_file(fileName):
    
    data = open(fileName).read().split('\n')
    
    targets = data[15::16] # extract just targets first
    targets = "".join(targets).split()
    targets = list(map(float, targets))
    targets = np.array(targets).reshape((2500,10))

    del data[::16]    # Now process data; first strip pattern name
    del data[14::15] # then strip targets
    
    data = "".join(data).split() # make it all into one gigantic string and split it into individual floats
    data = list(map(float, data)) # apply float to all data
    data = np.array(data).reshape((2500,196)) # convert long list into 2d array
    
    return data, targets

def create_targets_not2():

    x1 = np.append(np.zeros((1,500)), np.ones((1,250)))
    x2 = np.zeros((1,1750))
    x = np.append(x1, x2)
    return x.reshape(2500, 1)




def count_errors(results, threshold):

    correct = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(len(results)):
        if (i < 500 or i >= 750): # when not actually a 2
            if results[i] >= threshold:
                false_positives += 1
            else:
                correct += 1
        else: # when actually a 2
            if results[i] < threshold:
                false_negatives += 1
            else:
                correct += 1

    return correct, false_positives, false_negatives


# takes in csv file name and returns formatted 2d array (e.g. wih or who)
def read_csv(fileName):
    
    with open(fileName, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter="\n")
        lyst = []
        for row in csv_reader:
            entry = list(map(float, row[0].split()))
            lyst.append(entry)
    
    return np.array(lyst)

def report_results(p):
    results = p.test(testing_inputs)
    correct, false_positives, false_negatives = count_errors(results, 0.2)
    print("\nModel with {} iterations; {} hidden units; eta = {}".format(p._iters, p._h, p._eta))
    print("Overall accuracy: {}%".format(round((correct/len(results))*100, 4)))
    print("False positive rate: {}%".format(round((false_positives/len(results))*100, 4)))
    print("False negative rate: {}%".format(round((false_negatives/len(results))*100, 4)))


def print_confusion_matrix(p, inp):
    matrix = np.zeros((10,10))
    
    for j in range(len(inp)):
        target = j // 250
        _, _, _, _, Oj = p._forward(inp, j)

        output = np.argmax(Oj)
        
        matrix[target, output] += 1

    # I used Claude to do print formatting: https://claude.ai/chat/5ba262a6-b532-4572-bf25-c97fc582fd32
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
    for i, row in enumerate(matrix):
        print(f"{i:>15} |", end="")
        for val in row:
            print(f"{int(val):>{col_width}}", end="")
        print()

    return matrix

# Got code from Claude: https://claude.ai/chat/508f12be-e473-470d-a5e9-b4ff6fbecf41
def visualize_3d(matrix):
    n = matrix.shape[0]
    max_val = matrix.max()

    # --- Build bar positions ---
    xpos, ypos, zpos = [], [], []
    dx, dy, dz = [], [], []
    colors = []

    for row in range(n):
        for col in range(n):
            val = matrix[row, col]
            if val == 0:
                continue
            xpos.append(col)
            ypos.append(row)
            zpos.append(0)
            dx.append(0.7)
            dy.append(0.7)
            dz.append(val)

            if row == col:
                # Diagonal: green gradient
                t = val / max_val
                colors.append((0.1, 0.5 + 0.4 * t, 0.3 + 0.4 * t, 0.9))
            else:
                # Off-diagonal: red gradient
                t = min(val / 30, 1.0)
                colors.append((0.7 + 0.3 * t, 0.1, 0.1, 0.6 + 0.3 * t))

    # --- Plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, edgecolor='k', linewidth=0.2)

    # Axes labels
    ax.set_xlabel('Predicted Digit', labelpad=10)
    ax.set_ylabel('Actual Digit', labelpad=10)
    ax.set_zlabel('Count', labelpad=8)
    ax.set_title('Confusion Matrix — 3D Bar Chart', fontsize=14, pad=20)

    ax.set_xticks(np.arange(n) + 0.35)
    ax.set_xticklabels(range(n))
    ax.set_yticks(np.arange(n) + 0.35)
    ax.set_yticklabels(range(n))

    ax.view_init(elev=30, azim=-50)

    # --- Legend ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.1, 0.85, 0.6, 0.9), edgecolor='k', label='Correct (diagonal)'),
        Patch(facecolor=(0.9, 0.1, 0.1, 0.8), edgecolor='k', label='Misclassified'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 0.95))

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #plt.tight_layout()
    plt.savefig('confusion_matrix_3d_bar.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    print("–"*70)
    print("Part 1: XOR")
    print("\n10,000 iterations; 3 hidden units; eta = 0.5")
    
    inp = [[0,0], [0,1], [1,0], [1,1]]
    xor_tgt = [[0], [1], [1], [0]]

    p = Model(2,1,3) # for some reason, I sometimes got weird results, so I pickled good weights
    #p.train(inp, xor_tgt, iters=10000)
    #p.save("xor_weights.pkl")
    p.load("xor_weights.pkl")
    
    results = p.test(inp)
    for i in range(len(inp)):
        print("Input: {} -> Output: {}".format(inp[i], results[i]))

    print("–"*70)
    print("Part 2: 2-not-2")
    print("\nTesting a base model + 3 models with one variation each")
    
    training_inputs, targets_full = process_file('digits_train.txt')
    testing_inputs, _ = process_file('digits_test.txt')
    targets_not2 = create_targets_not2()

    # Training was done on separate computer, then brought over to my computer as csv file, then pickled here.
    # For example, Windows computer: pbase.train(training_inputs, targets_not2, iters=1000), then np.savetxt("2not2_petawih.csv", pbase.wih)
    # Go to my Mac: read_csv(), then save() and load as needed
    
    # Baseline model: 1000 iterations, 10 hidden units, eta = 0.5
    pbase = Model(196,1,10)
    pbase.load("2not2_pbase.pkl")

    # Model for increasing iterations: 2000 iterations, 10 hidden units, eta = 0.5
    piter = Model(196,1,10, iters = 2000) # this doesn't actually control iters here, it's just for printing purposes.
    # iters was controlled as 2000 on windows computer that I trained it with
    piter.load("2not2_piter.pkl")

    # Model for increasing number of hidden units: 1000 iterations, 20 hidden units, eta = 0.5
    phid = Model(196,1,20)
    phid.load("2not2_phid.pkl")

    # Model for decreasing eta to 0.2: 1000 iterations, 10 hidden units, eta = 0.2
    peta = Model(196,1,10, eta=0.2)
    peta.load("2not2_peta.pkl")

    models = [pbase, piter, phid, peta]
    for p in models:
        report_results(p)
    
    print("–"*70)
    print("Part 3: Full Monty\n")

    # Remember, we already have target_full from process_file above

    p3 = Model(196,10,50)
    #p3.train(training_inputs, targets_full, iters = 10000, eta = 0.3)
    #p3.wih = read_csv("full_wih.csv")
    #p3.who = read_csv("full_who.csv")
    #p3.save("full_classifier.pkl")
    p3.load("full_classifier.pkl")

    print("Model with {} iterations; {} hidden units; eta = {}\n".format(10000,50,0.3))
    confusion_matrix = print_confusion_matrix(p3, testing_inputs)
    print("\nOverall accuracy: {}%".format(round((confusion_matrix.trace()/len(testing_inputs))*100, 4)))
    print("–"*70)

    print("3D visualization using Claude:")
    visualize_3d(confusion_matrix)




























