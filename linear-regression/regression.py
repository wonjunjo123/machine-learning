
'''
Wonjun Jo
CSCI 315 : Artificial Intelligence
Instructor: Professor Simon Levy
Assignment #1
01/23/2026

Performs linear regression and classification on sample data.
After running script, you will see each graph sequentially for each part.
Exiting out of all graphs will yield the printed output for the assignment.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

# https://www.geeksforgeeks.org/python/import-text-files-into-numpy-arrays/
data = np.loadtxt("data.txt", skiprows=1)

n = data.shape[0] # number of rows

# Setting up variables
column_means = np.mean(data, axis=0) # contains the respective means of x1, x2, y, z
x1_mean = column_means[0]
x2_mean = column_means[1]
y_mean = column_means[2]
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]
z = data[:,3]

# ------------------------------- Part 1 -------------------------------

# Calculating weights for x1
m1_top = (x1 - x1_mean)*(y - y_mean)
m1_bottom = (x1 - x1_mean)**2
m1 = sum(m1_top)/sum(m1_bottom)
b1 = y_mean - m1*x1_mean

plt.plot(x1, y, 'o', label='x1', markersize=10)
plt.plot(x1, m1*x1 + b1, 'r', label='Fitted line')
plt.legend()
plt.title('Part 1 : x1 Individual')
plt.show()

# Calculating weights for x2
m2_top = (x2 - x2_mean)*(y - y_mean)
m2_bottom = (x2 - x2_mean)**2
m2 = sum(m2_top)/sum(m2_bottom)
b2 = y_mean - m2*x2_mean

plt.plot(x2, y, 'o', label='x2', markersize=10)
plt.plot(x2, m2*x2 + b2, 'r', label='Fitted line')
plt.legend()
plt.title('Part 1 : x2 Individual')
plt.show()

print("-"*60)
print("Part 1 : Individual Least Squares Solutions\n")
print("x1 : y = {}(x1) + {}".format(m1, b1))
print("x2 : y = {}(x2) + {}".format(m2, b2))
print("-"*60)

# ------------------------------- Part 2 -------------------------------

# Solving full linear regression
# Remember, we are treating b like a parameter, hence why we input an array of all 1's
A = np.vstack([x1, x2, np.ones(n)]).T
w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]

print("Part 2 : Full Regression Weights\n")
print("y = {}(x1) + {}(x2) + {}".format(w1, w2, b))
print("-"*60)


# https://stackoverflow.com/questions/36060933/plot-a-plane-and-points-in-3d-simultaneously
# https://www.geeksforgeeks.org/python/three-dimensional-plotting-in-python-using-matplotlib/
# https://chatgpt.com/c/69744d2b-85c4-832d-a03b-dc7984010cc1
def f(x1, x2):
    return w1*x1 + w2*x2 + b

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x1g = np.linspace(x1.min(), x1.max(), 30)
x2g = np.linspace(x2.min(), x2.max(), 30)

X1, X2 = np.meshgrid(x1g, x2g)
Z = f(X1, X2)
ax.plot_surface(X1, X2, Z, alpha=0.5, color="gray", edgecolor="none")#, shade=False)
ax.scatter(x1, y, zorder = 10)
ax.scatter(x2, y, zorder = 10)
ax.set_title('Part 2 : Full regression')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y_hat")
#ax.legend()
plt.show()


# ------------------------------- Part 3 -------------------------------

# Turning regression into classification
arr = w1*x1 + w2*x2 + b
classification = np.where(arr > 0, 1, 0)
# be careful because z has floating point 1's and 0's but classification has integer 1's and 0's
# but in numpy, I guess it type casts for me
validation = np.where(classification == z, 1, 0)
success_rate = np.average(validation)

print("Part 3 : Success Rate\n")
print("Success rate : {}%".format(success_rate*100))
print("-"*60)

# ------------------------------- Part 4 -------------------------------

print("Part 4 : Out of Sample Data Performance")

# the increments in which we will increase our sample size by
size_increment = 25

# a 2D array containing the weights from training on different sample sizes
weights = np.zeros(shape = (4,3), dtype = float)


for i in range(1, n//size_increment+1):
    # computing the weights for each training sample size
    sample_size = i*size_increment

    description = "\nBaseline Test" if (i == n//size_increment) else "\nTraining on first {} examples"
    print(description.format(sample_size))
    
    if (i < n//size_increment): # we don't need to compute weights for Baseline Test
        A = np.vstack([x1[0:sample_size], x2[0:sample_size], np.ones(sample_size)]).T
        weights[i-1][0], weights[i-1][1], weights[i-1][2] = np.linalg.lstsq(A, y[0:sample_size], rcond=None)[0]

    
    print("y = {}(x1) + {}(x2) + {}".format(weights[i-1][0], weights[i-1][1], weights[i-1][2]))

    # for baseline test, artificially set sample_size = 0 to make baseline test use all examples
    if (i==n//size_increment):
        sample_size = 0

    # now we apply those weights to the remaining "out of sample" data
    arr = (weights[i-1][0])*(x1[sample_size:]) + (weights[i-1][1])*(x2[sample_size:]) + weights[i-1][2]

    # we now convert regression value into classification value
    classification = np.where(arr > 0, 1, 0)

    # if the model correctly classified, we store 1
    validation = np.where(classification == z[sample_size:], 1, 0)
    success_rate = np.average(validation)
    
    print("Success rate on final {} examples: {}%".format(n-sample_size, success_rate*100))

print("-"*60)

#plt.show()


