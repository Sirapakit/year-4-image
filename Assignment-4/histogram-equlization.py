import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../low-contrast.jpg', 0)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.figure()
plt.hist(img.flatten(), 256, [0, 256])
plt.xlim([0, 256])

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]
plt.figure()
plt.hist(img2.flatten(), 256, [0, 256])
plt.xlim([0, 256])

cv.imshow('img', img)
cv.imshow('img2', img2)
cv.imwrite('low-contrast-2.jpg', img2)
plt.show()
cv.waitKey(0)
