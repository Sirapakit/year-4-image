import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Open the image
path = r'/Users/sirap/Documents/Year_4/year-4-image/shiba.jpeg'
img = cv.imread(path, 0)

m, n = img.shape

horizontal_kernel = np.array([[0, 0, 0], 
                              [0, 1, 0], 
                              [0, 0, -1]])  
vertical_kernel   = np.array([[0, 0, 0], 
                              [0, 0, 1], 
                              [0, -1, 0]])

new_image = np.zeros((m, n))

for i in range(1, m-1):
    for j in range(1, n-1):
        horizontal_gradient = (horizontal_kernel[0, 0] * img[i - 1, j - 1]) + \
                            (horizontal_kernel[0, 1] * img[i - 1, j]) + \
                            (horizontal_kernel[0, 2] * img[i - 1, j + 1]) + \
                            (horizontal_kernel[1, 0] * img[i, j - 1]) + \
                            (horizontal_kernel[1, 1] * img[i, j]) + \
                            (horizontal_kernel[1, 2] * img[i, j + 1]) + \
                            (horizontal_kernel[2, 0] * img[i + 1, j - 1]) + \
                            (horizontal_kernel[2, 1] * img[i + 1, j]) + \
                            (horizontal_kernel[2, 2] * img[i + 1, j + 1])

        vertical_gradient = (vertical_kernel[0, 0] * img[i - 1, j - 1]) + \
                            (vertical_kernel[0, 1] * img[i - 1, j]) + \
                            (vertical_kernel[0, 2] * img[i - 1, j + 1]) + \
                            (vertical_kernel[1, 0] * img[i, j - 1]) + \
                            (vertical_kernel[1, 1] * img[i, j]) + \
                            (vertical_kernel[1, 2] * img[i, j + 1]) + \
                            (vertical_kernel[2, 0] * img[i + 1, j - 1]) + \
                            (vertical_kernel[2, 1] * img[i + 1, j]) + \
                            (vertical_kernel[2, 2] * img[i + 1, j + 1])

        edge_magnitude = np.sqrt(pow(horizontal_gradient, 2.0) + pow(vertical_gradient, 2.0))
        new_image[i - 1, j - 1] = edge_magnitude


# plt.figure()
plt.imshow(new_image, cmap='gray')
plt.imsave('Sobel.png', new_image, cmap='gray', format='png')
plt.show()




# diagonal1_kernel = np.array([[0, 0, 0], 
#                               [0, 1, 0], 
#                               [0, 0, -1]]) 
# diagonal2_kernel   = np.array([[0, 0, 0], 
#                               [0, 0, 1], 
#                               [0, -1, 0]])