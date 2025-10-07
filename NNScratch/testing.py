import numpy as np

def correlate2d_valid(input_2d, kernel_2d):
    H_in, W_in = input_2d.shape
    H_k, W_k = kernel_2d.shape

    H_out = H_in - H_k + 1
    W_out = W_in - W_k + 1

    out = np.zeros((H_out, W_out), dtype=np.float32)

    for i in range(H_out):
        for j in range(W_out):
            # Element-wise multiply and sum
            region = input_2d[i:i+H_k, j:j+W_k]
            out[i, j] = np.sum(region * kernel_2d)

    return out


inp = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], dtype=np.float32)

kernel = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)

print(correlate2d_valid(inp, kernel))

