import numpy as np

inputs = [[4.8, 1.21, 2.385],
          [8.9, -1.81, 0.2],
          [1.41, 1.051, 0.026]]

exp_values = np.exp(inputs)

'''
    for one set or one batch, the below code does the normalization
norm_values = exp_values / np.sum(exp_values)
'''
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) 
print(norm_values)

'''
The issue with exp is that the number can get to infinite for a a value like 
x = 1000 and that would cause a overflow so we subtract the inputs or X with
the largest number of X then the maximum number would be a 0 and this will 
prevent us from having a overflow in the value and doing this would have
no affect on the output so the output remains the same relatively.
'''
