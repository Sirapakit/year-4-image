import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'../low-contrast.jpg'
image = cv2.imread(path)

# convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resized easier conv
dim = (200, 200)
image_rez = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
box_filter = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]]) # This is sharpen image

box_filter_5x5 = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],

])

gaussian_filter = np.array([
    [0.0625, 0.125, 0.0625],
    [0.0625, 0.125, 0.0625]
])

gaussian_filter_5x5 = (1/256) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

mexican_hat_filter = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

mexican_hat_filter_5x5 = (1/256) * np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
])

def calculate_output_image_size(img_size: int, kernel_size: int):
    num_pixels = 0
    for i in range(img_size):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1
    return num_pixels

def zero_padding(img: np.array, padding_width: int):
    img_zero_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2,  
        img.shape[1] + padding_width * 2
    ))
    img_zero_padding[padding_width:-padding_width, padding_width:-padding_width] = img
    return img_zero_padding

def convolve(img: np.array, kernel: np.array):
    tgt_size = calculate_output_image_size(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0]
    )
    k = kernel.shape[0]
    convolved_img = np.zeros(shape=(tgt_size, tgt_size))
    
    for i in range(tgt_size):
        for j in range(tgt_size):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
    return convolved_img

zero_padding(image_rez,1)

# Convulution 3 x 3
img_box = convolve(img=np.array(image_rez), kernel=box_filter)
img_gaussian = convolve(img=np.array(image_rez), kernel=gaussian_filter)
img_mexican_hat = convolve(img=np.array(image_rez), kernel=mexican_hat_filter)

# Convulution 5 x 5
img_box_5x5 = convolve(img=np.array(image_rez), kernel=box_filter_5x5)
img_gaussian_5x5 = convolve(img=np.array(image_rez), kernel=gaussian_filter_5x5)
img_mexican_hat_5x5 = convolve(img=np.array(image_rez), kernel=mexican_hat_filter_5x5)

# Subplot 1
fig, ax = plt.subplots(3, 2, figsize=(8, 8),constrained_layout=True)
fig.suptitle("Three types of filters", fontsize=16)

# plot Box Filter
ax[0, 0].set_title('Original')
ax[0, 0].imshow(image_rez)
ax[0, 1].set_title('Box Filter')
ax[0, 1].imshow(img_box)

# plot Gaussian Filter
ax[1, 0].set_title('Original')
ax[1, 0].imshow(image_rez)
ax[1, 1].set_title('Gaussian Filter')
ax[1, 1].imshow(img_gaussian)

# plot Mexican Hat Filter
ax[2, 0].set_title('Original')
ax[2, 0].imshow(image_rez)
ax[2, 1].set_title('Mexican Hat Filter')
ax[2, 1].imshow(img_mexican_hat)



# Subplot 2
fig, ax = plt.subplots(3, 2, figsize=(8, 8),constrained_layout=True)
fig.suptitle("3x3 VS 5x5", fontsize=16)

# plot Box Filter 5x5
ax[0, 0].set_title('Box Filter 3x3')
ax[0, 0].imshow(img_box)
ax[0, 1].set_title('Box Filter 5x5')
ax[0, 1].imshow(img_box_5x5)

# plot Gaussian Filter 5x5
ax[1, 0].set_title('Gaussian Filter 3x3')
ax[1, 0].imshow(img_gaussian)
ax[1, 1].set_title('Gaussian Filter 5x5')
ax[1, 1].imshow(img_gaussian_5x5)

# plot Mexican Hat Filter 5x5
ax[2, 0].set_title('Mexican Hat Filter 3x3')
ax[2, 0].imshow(img_mexican_hat_5x5)
ax[2, 1].set_title('Mexican Hat Filter 5x5')
ax[2, 1].imshow(img_mexican_hat_5x5)

plt.show()