import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# path = r'/Users/sirap/Documents/Year_4/year-4-image/shiba.jpeg'
path = r'/Users/sirap/Documents/Year_4/year-4-image/maxF.png'


img = cv.imread(path, 0)

                    
roberts_1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0,-1]]),
  
roberts_2 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])

m, n = img.shape
new_image = np.zeros((m, n))

for i in range(1, m-1):
    for j in range(1, n-1):
        horizontal_gradient = np.sum(np.multiply(roberts_1, img[i-1:i+2, j-1:j+2]))
        vertical_gradient = np.sum(np.multiply(roberts_2, img[i-1:i+2, j-1:j+2]))
        edge_magnitude = np.sqrt(pow(horizontal_gradient, 2.0) + pow(vertical_gradient, 2.0))
        new_image[i - 1, j - 1] = edge_magnitude

plt.imshow(new_image, cmap='gray')
plt.imsave('Robert_2.png', new_image, cmap='gray', format='png')
plt.show()