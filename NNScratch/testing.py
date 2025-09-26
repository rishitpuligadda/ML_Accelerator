import numpy as np 
import nnfs
from NeuralNetwork import Activation_Softmax_Loss_CategoricalCrossentropy, Activation_SoftMax, Loss_CategoricalCrossEntropy

#nnfs.init()

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)

dvalues1 = softmax_loss.dinputs

activation = Activation_SoftMax()
activation.output = softmax_outputs

loss = Loss_CategoricalCrossEntropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)

dvalues2 =  activation.dinputs

print("blahhhhh")
print(dvalues1)
print()
print(dvalues2)
