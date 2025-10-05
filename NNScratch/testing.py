import numpy as np

def read_txt(filename):
    """Read a whitespace-separated txt file into a numpy array."""
    return np.loadtxt(filename)

def compute_fixed_point_params(weights_file, bias_file, input_file=None, frac_precision=12):
    """
    Compute required WIDTH and FRAC for a layer.

    weights_file : str  : path to layer weights (2D)
    bias_file    : str  : path to layer biases (1D)
    input_file   : str  : path to input vector (1D or 2D, optional)
    frac_precision : int : desired fractional bits for precision
    
    Returns:
        dict with max sum, suggested integer bits, total WIDTH, FRAC
    """
    W = read_txt(weights_file)
    B = read_txt(bias_file)
    
    if input_file:
        X = read_txt(input_file)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        max_input = np.max(np.abs(X))
    else:
        max_input = 1.0  # assume normalized input if not provided

    max_weight = np.max(np.abs(W))
    max_bias   = np.max(np.abs(B))
    
    N = W.shape[1]  # number of inputs to this layer
    sum_max = N * max_weight * max_input + max_bias
    
    # Fractional bits
    FRAC = frac_precision
    
    # Scale max sum
    sum_scaled = sum_max * (2 ** FRAC)
    
    # Integer bits required
    INT_BITS = int(np.ceil(np.log2(sum_scaled))) + 1  # +1 for sign
    TOTAL_WIDTH = INT_BITS + FRAC
    
    return {
        "max_input": max_input,
        "max_weight": max_weight,
        "max_bias": max_bias,
        "sum_max": sum_max,
        "FRAC": FRAC,
        "INT_BITS": INT_BITS,
        "TOTAL_WIDTH": TOTAL_WIDTH
    }

if __name__ == "__main__":
    # Example usage for layer1
    layer1_params = compute_fixed_point_params(
        weights_file="../parameters/weights_layer1.txt",
        bias_file="../parameters/biases_layer1.txt",
        input_file="../parameters/input_image.txt",
        frac_precision=12
    )
    
    print("Layer 1 fixed-point parameters:")
    for k, v in layer1_params.items():
        print(f"{k}: {v}")

    # You can repeat for layer2 and layer3
    layer2_params = compute_fixed_point_params(
        weights_file="weights_layer2.txt",
        bias_file="biases_layer2.txt",
        input_file="None",
        frac_precision=12
    )
    print("\nLayer 2 fixed-point parameters:")
    for k, v in layer2_params.items():
        print(f"{k}: {v}")

    layer3_params = compute_fixed_point_params(
        weights_file="weights_layer3.txt",
        bias_file="biases_layer3.txt",
        input_file="None",
        frac_precision=12
    )
    print("\nLayer 3 fixed-point parameters:")
    for k, v in layer3_params.items():
        print(f"{k}: {v}")

