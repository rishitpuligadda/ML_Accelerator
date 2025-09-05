import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
Dot product of the elements without using numpy
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = neuron_bias
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += weight*n_input
    layer_outputs.append(neuron_output)
print(layer_outputs)
'''

#the below code is the same as the above
'''
the shape of weights is (3, 4) and shape of inputs is (4,)
    so if they are interchanged in the below code then it is
    (4,) * (3, 4) which is invalid in terms of matrix multiplication
    thus it will throw a invalid shape error
    interchange i mean
    np.dot(inputs, weights)
'''
output = np.dot(weights, inputs) + biases
print(output)
