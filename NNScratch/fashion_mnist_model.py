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

fashion_mnist_labels = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
                        }

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
        optimizer = Optimizer_Adam(decay = 1e-4),
        accuracy = Accuracy_Categorical()
        )

model.finalize()

model.train(X, y, validation_data = (X_test, y_test), epochs = 10, batch_size = 128, print_every = 100)

parameters = model.get_parameters()
model.save_parameters()

confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)

for prediction in predictions:
    print(fashion_mnist_labels[prediction])

model.save('fashion_mnist.model')

image_data = cv2.imread('../fashion_mnist_images/test/0/0000.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

np.savetxt('../parameters/input_image.txt', image_data, fmt = "%.6f")

confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)

