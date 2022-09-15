import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


path = r'image1.jpg'

# Bgr to grayscle using cv
imageCV = cv.imread(path)
# cv.imshow('img1', imageCV)
grayscale_cv = cv.cvtColor(imageCV, cv.COLOR_BGR2GRAY)
# cv.imshow('img2', grayscale_cv)
cv.waitKey(0)
cv.destroyAllWindows()

# Bgr to grayscle using Matplotlib
imageMAT = mpimg.imread(path)
Red = imageMAT[:, :, 0]
Green = imageMAT[:, :, 1]
Blue = imageMAT[:, :, 2]
imgGray = 0.2989 * Red + 0.5870 * Green + 0.1140 * Blue
plt.imshow(imgGray, cmap='gray')
plt.show()

# Create histogram from those two image
plt.subplot(121)
hist, bin = np.histogram(grayscale_cv.ravel(), 256, [0, 255])
plt.plot(hist)
plt.subplot(122)
hist, bin = np.histogram(imgGray.ravel(), 256, [0, 255])
plt.plot(hist)
plt.show()

# Create histogram using pixel
row_cv, col_cv = grayscale_cv.shape[0], grayscale_cv.shape[1]
y1 = np.zeros((256), np.uint64)
for i in range(row_cv):
    for j in range(col_cv):
        y1[grayscale_cv[i, j]] = y1[grayscale_cv[i, j]] + 1
x = np.arange(0, 256)
plt.subplot(211)
plt.plot(x, y1)

row_mat, col_mat = imageMAT.shape[0], imageMAT.shape[1]
y2 = np.zeros((256), np.uint64)
for i in range(row_mat):
    for j in range(col_mat):
        y2[imageMAT[i, j]] = y2[imageMAT[i, j]] + 1
x = np.arange(0, 256)
plt.subplot(212)
plt.plot(x, y2)
plt.show()
