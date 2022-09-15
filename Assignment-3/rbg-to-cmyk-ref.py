import numpy as np
import matplotlib.pyplot as plt
import cv2

path = r'../darth.jpg'

# RGB to CMYK using openCV
image = cv2.imread(path)

# Create float
bgr = image.astype(np.float)/255.0

# Extract channels
with np.errstate(invalid='ignore', divide='ignore'):
    K = 1 - np.max(bgr, axis=2)
    C = (1-bgr[:, :, 2] - K)/(1-K)
    M = (1-bgr[:, :, 1] - K)/(1-K)
    Y = (1-bgr[:, :, 0] - K)/(1-K)

# Convert the input BGR image to CMYK colorspace
CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)

cv2.imshow('sample', image)
plt.imshow(CMYK)
cv2.imwrite('CMYK.tiff', CMYK)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
