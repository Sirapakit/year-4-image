import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

path = r'/Users/sirap/Documents/Year_4/year-4-image/shiba.jpeg'
img = cv.imread(path,0)
edges = cv.Canny(img,100,200)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(edges,cmap = 'gray')
plt.title('Canny')

plt.show()