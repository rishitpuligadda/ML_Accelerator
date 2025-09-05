import numpy as np
'''
batch of inputs is nothing but how many inputs are run through
the neuron at once. Here we are running 3 at once and we are 
taking the transpose of the weights here because size of input
is (3,4) and the size of weights is (3,4) which is not a valid
matrix multiplication so we do (4,3) and then it becomes a 
valid multiplication and also we are multiplying the correct
values with one and other to get the output.
1 * 0.2 + 2 * 0.8 - 3 * 0.5 + 2.5 * 1
'''
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases

#Adding one more layer where the input to that layer is the above

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer2_output)
