import numpy as np
import cv2 as cv
from statistics import median
from itertools import repeat

# path = r'../s_and_p_noise.png'
path = r'../maxF.png'
img = cv.imread(path, 0)
cv.imshow("Image", img)
m, n = img.shape
img_reduce_noise = np.zeros([m, n])

weighted_matrix = np.array([[3,2,3],
                            [2,1,2],
                            [3,2,3]])
weighted_list = weighted_matrix.flatten().tolist()
print(weighted_list[0])
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = []
        temp.extend(repeat(img[i-1, j-1],weighted_list[0]))
        temp.extend(repeat(img[i-1, j],weighted_list[1]))
        temp.extend(repeat(img[i-1, j+1],weighted_list[2]))
        temp.extend(repeat(img[i, j-1],weighted_list[3]))
        temp.extend(repeat(img[i, j],weighted_list[4]))
        temp.extend(repeat(img[i, j+1],weighted_list[5]))
        temp.extend(repeat(img[i+1, j-1],weighted_list[6]))
        temp.extend(repeat(img[i+1, j],weighted_list[7]))
        temp.extend(repeat(img[i+1, j+1],weighted_list[8]))
        img_reduce_noise[i, j]= median(temp)
img_reduce_noise = img_reduce_noise.astype(np.uint8)
cv.imshow('new_weighted_median_filtered', img_reduce_noise)
cv.waitKey(0)




