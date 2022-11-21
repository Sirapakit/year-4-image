import numpy as np
import matplotlib.pyplot as plt
import cv2 

# path = r'/Users/sirap/Documents/Year_4/year-4-image/low-contrast.jpg' 
path = r'/Users/sirap/Documents/Year_4/year-4-image/shiba.jpeg'
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dim = (400, 400)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

sharpen_kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

gaussian_kernel = (1/256) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
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

# Sharp
img_sharp = convolve(img=np.array(image), kernel=sharpen_kernel)
img_sharp = zero_padding(img_sharp, 1 )

# UnSharp
gaussian = convolve(img=np.array(image), kernel=gaussian_kernel)
gaussian = zero_padding(gaussian, 2 )
mask =  image - gaussian 
img_unsharp = image + ( mask * 0.2 )

fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout=True)
fig.suptitle("Sharp vs Unsharp", fontsize=12)

# plot Sharpen Filter
ax[0, 0].set_title('Original')
ax[0, 0].imshow(image, cmap="gray")
ax[0, 1].set_title('Shapen Filter')
ax[0, 1].imshow(img_sharp, cmap="gray")

# plot Unsharp Filter
ax[1, 0].set_title('Original')
ax[1, 0].imshow(image, cmap="gray")
ax[1, 1].set_title('UnSharpen Filter')
ax[1, 1].imshow(img_unsharp, cmap="gray")

plt.show()





