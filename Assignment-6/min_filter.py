import numpy as np
import cv2 as cv

# path = r'../s_and_p_noise.png'
path = r'../maxF.png'
img = cv.imread(path, 0)
cv.imshow("Image", img)

m, n = img.shape
img_min = np.zeros([m, n])

for i in range(1, m-1):
    for j in range(1, n-1):

        temp = [img[i-1, j-1],img[i-1, j],img[i-1, j + 1],
               img[i, j-1],img[i, j],img[i, j + 1],
               img[i + 1, j-1],img[i + 1, j],img[i + 1, j + 1]]

        img_min[i, j]= min(temp)

img_min = img_min.astype(np.uint8)
cv.imshow('new_min_filtered', img_min)

cv.waitKey(0)