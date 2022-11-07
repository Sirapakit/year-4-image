import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Open the image
path = r'/Users/sirap/Documents/Year_4/year-4-image/tony.jpeg'

img = cv.imread(path, 0)

horizontal_kernel = np.array([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]])  
vertical_kernel   = np.array([[-1, -2, -1], 
                              [0, 0, 0], 
                              [1, 2, 1]])  

m, n = img.shape
new_image = np.zeros((m, n))

for i in range(1, m-1):
    for j in range(1, n-1):
        horizontal_gradient = np.sum(np.multiply(horizontal_kernel, img[i-1:i+2, j-1:j+2]))
        vertical_gradient = np.sum(np.multiply(vertical_kernel, img[i-1:i+2, j-1:j+2]))
        # edge_magnitude = np.sqrt(pow(horizontal_gradient, 2.0) + pow(vertical_gradient, 2.0))
        edge_magnitude = np.sqrt(pow(horizontal_gradient, 2.0) )
        new_image[i - 1, j - 1] = edge_magnitude

plt.imshow(new_image, cmap='gray')
plt.imsave('Sobel.png', new_image, cmap='gray', format='png')
plt.show()

