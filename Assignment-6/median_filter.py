import numpy as np
import cv2 as cv
from statistics import median

path = r'../s_and_p_noise.png'
# path = r'../maxF.png'
img = cv.imread(path, 0)
cv.imshow("Image", img)

m, n = img.shape
img_reduce_noise = np.zeros([m, n])

for i in range(1, m-1):
    for j in range(1, n-1):

        temp = [img[i-1, j-1],img[i-1, j],img[i-1, j + 1],
               img[i, j-1],img[i, j],img[i, j + 1],
               img[i + 1, j-1],img[i + 1, j],img[i + 1, j + 1]]

        img_reduce_noise[i, j]= median(temp)

img_reduce_noise = img_reduce_noise.astype(np.uint8)
cv.imshow('new_median_filtered', img_reduce_noise)

cv.waitKey(0)

# A = np.array([[15,17,28],
# [31,62,61],
# [50,48,57]])
# print(A.shape)
# print(A.ndim)

# B = A.flatten()
# # median = np.median(B)
# # A[1][1] = median
# print(B.shape)
# print(B.ndim)


