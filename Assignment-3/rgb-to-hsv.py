import cv2
import numpy as np
import matplotlib.pyplot as plt
# RGB to HSV
path = r'../darth.jpg'
image = cv2.imread(path)

# use cv2 library
RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
HSVimage = cv2.cvtColor(RGBimage, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV CV2', HSVimage)


# use formula
def rgb2hsv_np(img_rgb):
    assert img_rgb.dtype == np.float64

    height, width, c = img_rgb.shape
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    t = np.min(img_rgb, axis=-1)
    v = np.max(img_rgb, axis=-1)

    s = (v - t) / (v + 1e-6)
    s[v == 0] = 0

    hr = 60 * (g - b) / (v - t + 1e-6)
    hg = 120 + 60 * (b - r) / (v - t + 1e-6)
    hb = 240 + 60 * (r - g) / (v - t + 1e-6)

    h = np.zeros((height, width), np.float32)

    h = h.flatten()
    hr = hr.flatten()
    hg = hg.flatten()
    hb = hb.flatten()

    h[(v == r).flatten()] = hr[(v == r).flatten()]
    h[(v == b).flatten()] = hb[(v == b).flatten()]
    h[(v == g).flatten()] = hg[(v == g).flatten()]

    h[h < 0] += 360
    h = h.reshape((height, width))

    img_hsv = np.stack([h, s, v], axis=-1)
    return img_hsv


img_rgb = image / 255.0
img_rgb = img_rgb.astype(np.float64)

img_hsv1 = rgb2hsv_np(img_rgb)

cv2.imshow('hsv-formula', img_hsv1)
cv2.waitKey(0)
cv2.destroyAllWindows

# https://stackoverflow.com/questions/63691352/rgb-to-hsv-in-numpy/
# https://stackoverflow.com/questions/63835007/hsv-to-rgb-in-numpy
