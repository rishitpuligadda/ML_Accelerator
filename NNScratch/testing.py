import cv2 
import matplotlib.pyplot as plt

image_data = cv2.imread('../fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)

plt.imshow(image_data)
plt.savefig('Plot.png')
