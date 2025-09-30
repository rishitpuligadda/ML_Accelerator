import numpy as np 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_SoftMax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Create a layer: 4 inputs → 3 neurons
layer1 = Layer_Dense(4, 3)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3, 2)
activation2 = Activation_SoftMax()

# Example input batch: 2 samples, each with 4 features
inputs = np.array([
    [1, 2, 3, 4],
    [2, -1, 0, 3]
])

# Forward pass
layer1.forward(inputs)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

print("Softmax Output:")
print(activation2.output)

np.savetxt("../parameters/outputlayer1.txt", layer1.output, fmt="%.6f")
np.savetxt("../parameters/inputs.txt", inputs, fmt="%.6f")


# Get weights and biases from the layers
weights1 = layer1.weights.T
biases1  = layer1.biases
weights2 = layer2.weights.T
biases2  = layer2.biases

# ---- Save biases, one per line ----
with open("../parameters/biases_layer1.txt", "w") as f:
    for val in biases1.flatten():  # Flatten to 1D
        f.write(f"{val:.6f}\n")    # Write each bias on a separate line

with open("../parameters/biases_layer2.txt", "w") as f:
    for val in biases2.flatten():
        f.write(f"{val:.6f}\n")    # Write each bias on a separate line

# ---- Save weights row by row ----
with open("../parameters/weights_layer1.txt", "w") as f:
    for row in weights1:
        line = " ".join(f"{val:.6f}" for val in row)
        f.write(line + "\n")

with open("../parameters/weights_layer2.txt", "w") as f:
    for row in weights2:
        line = " ".join(f"{val:.6f}" for val in row)
        f.write(line + "\n")

print("All weights and biases are stored in the files, biases one per line.")
