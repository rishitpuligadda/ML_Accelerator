import numpy as np
import cv2

# ---- Load and preprocess the image ----
image_data = cv2.imread('../fashion_mnist_images/test/0/0002.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data  # invert colors if needed
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5  # normalize [-1,1]

# ---- Save as floating-point text ----
np.savetxt('../parameters/input_image.txt', image_data, fmt="%.6f")

print("input_image.txt generated successfully!")

