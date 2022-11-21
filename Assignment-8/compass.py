import numpy as np
import matplotlib.pyplot as plt
import cv2 

path = r'/Users/sirap/Documents/Year_4/year-4-image/shiba.jpeg'
img = cv2.imread(path, 0)

n_kernel = np.array([[-1, 0, 1], 
                    [-1, 0, 1], 
                    [-1, 0, 1]])  

ne_kernel = np.array([[-1, -1, 0], 
                    [-1, 0, 1], 
                    [0, 1, 1]])  

e_kernel = np.array([[-1, -1, -1], 
                    [0, 0, 0], 
                    [1, 1, 1]])  

se_kernel = np.array([[0, -1, -1], 
                    [1, 0, -1], 
                    [1, 1, 0]])  

s_kernel = -(n_kernel)
sw_kernel = -(ne_kernel)
w_kernel = -(e_kernel)
nw_kernel = -(se_kernel)

m, n = img.shape
new_image = np.zeros((m, n))

for i in range(1, m-1):
    for j in range(1, n-1):
        n_gradient = np.abs(np.sum(np.multiply(n_kernel, img[i-1:i+2, j-1:j+2])))
        ne_gradient = np.abs(np.sum(np.multiply(ne_kernel, img[i-1:i+2, j-1:j+2])))
        e_gradient = np.abs(np.sum(np.multiply(e_kernel, img[i-1:i+2, j-1:j+2])))
        se_gradient = np.abs(np.sum(np.multiply(se_kernel, img[i-1:i+2, j-1:j+2])))
        s_gradient = np.abs(np.sum(np.multiply(s_kernel, img[i-1:i+2, j-1:j+2])))
        sw_gradient = np.abs(np.sum(np.multiply(sw_kernel, img[i-1:i+2, j-1:j+2])))
        w_gradient = np.abs(np.sum(np.multiply(w_kernel, img[i-1:i+2, j-1:j+2])))
        nw_gradient = np.abs(np.sum(np.multiply(nw_kernel, img[i-1:i+2, j-1:j+2])))

        edge_magnitude = np.maximum(n_gradient, np.maximum(ne_gradient, np.maximum(e_gradient, np.maximum(se_gradient, np.maximum(s_gradient, np.maximum(sw_gradient, np.maximum(w_gradient, nw_gradient)))))))
        new_image[i-1:i+2, j-1:j+2] = edge_magnitude

fig, ax = plt.subplots(2, figsize=(8, 8),constrained_layout=True)
fig.suptitle("Original VS Compass", fontsize=12)

ax[0].set_title('Original')
ax[0].imshow(img, cmap="gray")
ax[1].set_title('Compass Filter')
ax[1].imshow(new_image, cmap="gray")

plt.show()

