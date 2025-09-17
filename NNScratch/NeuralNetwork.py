import numpy as np 

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if (len(y_true.shape)) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif (len(y_true.shape)) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        Totloss = -np.log(correct_confidences)
        return Totloss

layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3, 3)
activation2 = Activation_SoftMax()
loss_function = Loss_CategoricalCrossEntropy()
    
lowest_loss = 9999999
best_layer1_weights = layer1.weights.copy()
best_layer2_weights = layer2.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_biases = layer2.biases.copy()

#Training by the second method in optimization.py code
for i in range(10000):
    layer1.weights += 0.05 * np.random.randn(2, 3)
    layer1.biases += 0.05 * np.random.randn(1, 3)
    layer2.weights += 0.05 * np.random.randn(3, 3)
    layer2.biases += 0.05 * np.random.randn(1, 3)

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print(f"iteration: {i}, loss: {loss} & acc: {accuracy}")
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()
        lowest_loss = loss
    else:
        layer1.weights = best_layer1_weights.copy()
        layer1.biases = best_layer1_biases.copy()
        layer2.weights = best_layer2_weights.copy()
        layer2.biases = best_layer2_biases.copy()


print(best_layer1_weights, '\n')
print(best_layer2_weights, '\n')
print(best_layer1_biases, '\n')
print(best_layer2_biases)
