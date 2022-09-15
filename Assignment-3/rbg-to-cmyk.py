import cv2
import numpy as np
from matplotlib.image import imread

RGB_SCALE = 255
CMYK_SCALE = 100

path = r'C:/Users/sirap/Desktop/year-4-image/darths.jpg'

# RGB to CMYK using openCV
image = cv2.imread(path)

def splitRGB(image):
    b, g, r = cv2.split(image)
    return b, g, r

def rgbToCmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, CMYK_SCALE
    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    # return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE
    return (c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE)

def imageRgbToCmyk(image):
    height, width, _ = image.shape
    for i in range(height):
        for j in range(width):
            # Change pixels [i,j] from (b,g,r) to (C,M,Y,K)
            b, g, r = cv2.split(image)
            image[i, j] = rgbToCmyk(b[i, j], g[i, j], r[i, j])
        print("One row passes %d", i*j)
    # print(height, width)
    cv2.imshow('sample2', image)



imageRgbToCmyk(image)

cv2.imshow('sample', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
