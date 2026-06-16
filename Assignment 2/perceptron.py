import numpy as np

from tqdm import tqdm

class Perceptron:
    def __init__(self, n, m): # we are passing in rows and columns
        self.w = (np.random.randn(n+1, m)) / 1000 # this is creating the initialized values with normal distribution
	# but if you do it with np.random.random((n+1, m))*2 - 1, we would have uniform distribution
	# the assignment does np.random.random but Levy prefers to use normal distribution
	# I should only contain weights in the initialization!
	# the models are the weights themselves

    def __str__(self):
        return 'Perceptron with {} inputs and {} outputs'.format(self.w.shape[0]-1, self.w.shape[1])

    def _forward(self, Inputs, j): # the _ in the front is to indicate a private method
        Ij = np.append([1], Inputs[j])
        Oj = (np.dot(Ij, self.w) > 0).astype(int)
        return Ij, Oj

    def train(self, Inputs, Targets, iters=1000):
        p = len(Inputs) # number of patterns
        #for i in tqdm(range(iters)):
        for i in range(iters):
            dw = np.zeros(self.w.shape) # weight changes
            for j in range(p): # loop over patterns
                # "forward pass"
                Ij, Oj = self._forward(Inputs, j)
                Dj = Targets[j] - Oj
                dw += np.outer(Ij, Dj) # delta rule
                # instead of doing transpose and inner product, I think we are doing outer product so that we don't have to transpose
                self.w += dw/p
                '''
                Perceptron-learning algorithm from AIMA (2003)
                '''
                
    def test(self, Inputs, n=1):

        p = len(Inputs)
        results = np.zeros((p,n))
        for j in range(p):
            _, Oj = self._forward(Inputs, j)
            results[j] = Oj

        return results


def show_pattern(pattern):
    for row in range(14):
        for col in range(14):
            print("*" if pattern[14*row+col] > 0 else ' ', end = ' ')
        print()

        
# takes in data and converts into 2d array format for perceptron
# data parameter needs to be a "list of newlines" format
def process_data(data, dim):
    
    data = "".join(data).split() # make it all into one gigantic string and split it into individual floats
    data = list(map(float, data)) # apply float to all data
    data = np.array(data).reshape(dim) # convert long list into 2d array
    
    return data


if __name__ == "__main__":

    # format the inputs for training (and also optionally the original targets)
    data = open('digits_train.txt').read().split('\n')
    #targets = data[15::16] # extract just targets first
    #targets = process_data(targets, (2500,10))
    del data[::16]    # Strip pattern name
    del data[14::15]    # Strip pattern label (the targets)
    training_inputs = process_data(data, (2500,196))
    

    # format the inputs for testing
    test_data = open('digits_test.txt').read().split('\n')
    del test_data[::16]    # Strip pattern name
    del test_data[14::15]    # Strip pattern label (the targets)
    test_inputs = process_data(test_data, (2500,196))

    # ---------------------------- Part 3 ----------------------------
    # we are artifically creating the targets for 2 and not-2.
    #1 means it is 2, and 0 means it is not-2
    x1 = np.zeros((1,500))
    x2 = np.ones((1,250))
    x2 = np.append(x1, x2)
    x3 = np.zeros((1,1750))
    x = np.append(x2, x3)
    targets_not2 = x.reshape(2500, 1)

    #show_pattern(inputs[2499])
    
    ''' Testing simple cases for AND and OR
    # General boolean input
    inp = [[0,0], [0,1], [1,0], [1,1]]
    # Targets for AND
    and_tgt = [[0], [0], [0], [1]]
    or_tgt = [[0],[1],[1],[1]]
    '''
    
    #p = Perceptron(196,10)
    p_2 = Perceptron(196,1)
    #print(p.w)

    # train the model
    iterations = 1000
    p_2.train(training_inputs, targets_not2, iters = iterations)

    # test and store results
    results = p_2.test(test_inputs)

    # compare results to what they should be
    comp = (targets_not2 == results)
    correct_count = len(comp[comp == True])
    not2 = (targets_not2 != 1)
    report_2 = len(results[results == [1]]) # number of things predicted to be 2
    report_not2 = len(results[results == [0]]) # number of things predicted to not be 2

    false_positive = ~comp & not2 # was incorrect and it wasn't a two
    fp_count = len(false_positive[false_positive == True])

    false_negative = ~comp & ~not2 # was in correct and was a two
    fn_count = len(false_negative[false_negative == True])
    
    total = 2500
    false_count = total - correct_count
    accuracy = (correct_count / total)*100
    fp_rate = (fp_count / report_2)*100
    fn_rate = (fn_count / report_not2)*100
    print("\nTrain to discriminate 2 from not 2")
    print("Overall Performance with {} iterations:".format(iterations), end = " ") 
    print(f'{accuracy:.3f}%'.format(accuracy))
    print("False Positive Rate: {}%".format(fn_rate))
    print("False Negative Rate: {}%".format(fp_rate))


    # ---------------------------- Part 4 ----------------------------
    training_inputs_8 = np.append(training_inputs[:250], training_inputs[2000:2250])
    training_inputs_8 = training_inputs_8.reshape(500,196)

    # used Claude to figure out tile()
    # https://claude.ai/chat/47968da2-3f75-40de-9829-0748614a6293
    targets_8 = (np.append(np.tile([1,0], (250, 1)), np.tile([0,1], (250,1)))).reshape(500,2)

    test_inputs_8 = np.append(test_inputs[:250], test_inputs[2000:2250])
    test_inputs_8 = test_inputs_8.reshape(500,196)
    
    p_8 = Perceptron(196, 2)
    p_8.train(training_inputs_8, targets_8, iters = iterations)
    results1 = p_8.test(test_inputs_8, n=2)
    comp1 = (targets_8 == results1)
    correct_count1 = len(comp1[comp1 == True])/2
    is8 = (targets_8 == [0,1])
    report_8 = len(results1[results1 == [0,1]]) # number of things predicted to be 8
    report_0 = len(results1[results1 == [1,0]]) # number of things predicted to be 0

    false_positive = ~comp1 & ~is8 # was incorrect and it wasn't a two
    fp_count = len(false_positive[false_positive == True])

    false_negative = ~comp1 & is8 # was in correct and was a two
    fn_count = len(false_negative[false_negative == True])
    
    total = 500
    false_count = total - correct_count1
    accuracy1 = (correct_count1 / total)*100
    fp_rate = (fp_count / report_8)*100
    fn_rate = (fn_count / report_0)*100
    print("\nTrain to discriminate 8 from 0")
    print("Overall Performance with {} iterations:".format(iterations), end = " ") 
    print(f'{accuracy1:.3f}%'.format(accuracy1))
    print("False Positive Rate: {}%".format(fp_rate))
    print("False Negative Rate: {}%".format(fn_rate))

