import cv2
import numpy as np

# image is loaded with imread command
image = cv2.imread('../image1.jpg')

# convert the image in grayscale
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#  different thresholding
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)

cv2.imshow('be4 Thres', img)

# manual thres
height, width = img.shape
for i in range(height):
    for j in range(width):
        if np.sum(img[i, j]) <= 200:
            img[i, j] = 0
cv2.imshow('Thres', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
