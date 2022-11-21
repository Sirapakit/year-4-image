import cv2
import numpy as np
from matplotlib import pyplot as plt

path = r'/Users/sirap/Documents/Year_4/year-4-image/image1.jpg'
img = cv2.imread(path, 0)
img_ori = cv2.imread(path, 0)

edges = cv2.Canny(img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

for r_theta in lines:
	arr = np.array(r_theta[0], dtype=np.float64)
	r, theta = arr
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*r
	y0 = b*r
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

fig, ax = plt.subplots(2, figsize=(8, 8),constrained_layout=True)
fig.suptitle("Hough Transform", fontsize=12)

ax[0].set_title('Original')
ax[0].imshow(img_ori, cmap='gray')
ax[1].set_title('Hough')
ax[1].imshow(img, cmap='gray')

plt.show()

