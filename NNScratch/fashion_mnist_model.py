import numpy as np
import cv2 
import os
from NeuralNetwork import Layer_Dense, Activation_ReLU, Activation_SoftMax, Model, Loss_CategoricalCrossEntropy, Optimizer_Adam, Accuracy_Categorical

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('../fashion_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_SoftMax())

model.set(
        loss = Loss_CategoricalCrossEntropy(),
        optimizer = Optimizer_Adam(decay = 1e-3),
        accuracy = Accuracy_Categorical()
        )

model.finalize()

model.train(X, y, validation_data = (X_test, y_test), epochs = 10, batch_size = 128, print_every = 100)

model.evaluate(X, y)
