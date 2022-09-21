import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

path = r'../darth.jpg'
image = cv2.imread(path)

# use cv2
cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow('sample', cv2_image)


# use formula
B = image[:, :, 0]
G = image[:, :, 1]
R = image[:, :, 2]

X = 0.412453*R + 0.357580*G + 0.180423*B
Y = 0.212671*R + 0.715160*G + 0.0072169*B
Z = 0.019334*R + 0.119193*G + 0.950227*B

X = X / 0.960456
Z = Z / 1.088754

L = 903.3 * Y if Y.any() <= 0.008856 else (116 * (np.power(Y, 1/3))) - 16
L = L * 100 / 255


def calc(t):
    return 7.767*t + 16/116 if t.any() <= 0.008856 else np.power(t, 1/3)


delta = 128  # for 8-bit image
a = 500*((calc(X)) - (calc(Y))) + delta
b = 200*((calc(Y)) - (calc(Z))) + delta

img_lab = np.dstack((L, a, b)).astype(np.uint8)

print("L is %f", L)
print("a is %f", a)
print("b is %f", b)

cv2.imshow('test', img_lab)
plt.imshow(img_lab)
plt.show()
cv2.waitKey(0)
