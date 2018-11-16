import numpy as np
import network
import network2
import pdb  

def vectorized_result(j):
    """Return a 3-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...3) into a corresponding desired output from the neural
    network."""
    e = np.zeros((3, 1))
    e[j] = 1.0
    return e

data=np.load('../../train_test_data_II_III.npy')
training_data = data[0] 
training_inputs = training_data[0]
training_results = training_data[1]

test_data = data[1] 
test_inputs = test_data[0]
test_results = test_data[1]

# Bit of data reshaping required here for input into the FFNN from the
# network and nework2 packages from http://neuralnetworksanddeeplearning.com

training_inputs = [np.reshape(x, (7100, 1)) for x in training_inputs]
training_results = [vectorized_result(y) for y in training_results]
training_data = list(zip(training_inputs, training_results))

test_inputs = [np.reshape(x, (7100, 1)) for x in test_inputs]
test_data = list(zip(test_inputs, test_results))


# Use the standard mean square error.
net = network.Network([7100, 30, 3]) 
net.SGD(training_data, 40, 25, 1.0, test_data=test_data)


# Use of cross entropy and reularising paramater.
net = network2.Network([7100, 30, 3], cost=network2.CrossEntropyCost)
net.large_weight_initializer()

pdb.set_trace()
net.SGD(training_data, 100, 25, 0.5,
    evaluation_data=test_data,
    lmbda = 0.5, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)
