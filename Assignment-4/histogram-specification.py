import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import cv2

# reading main image
img_1 = cv2.imread("../low-contrast.jpg")
img1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

# checking the number of channels
print(f'No of Channel is: {str(img1.ndim)}')


# reading reference image
img_2 = cv2.imread("../low-contrast-2.jpg")
img2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# checking the number of channels
print(f'No of Channel is: {str(img2.ndim)}')

image = img1
reference = img2

matched = match_histograms(image, reference,
                           multichannel=True)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    figsize=(8, 3),
                                    sharex=True, sharey=True)

for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(image, cmap='gray')
ax1.set_title('Source')
ax2.imshow(reference, cmap='gray')
ax2.set_title('Reference')
ax3.imshow(matched, cmap='gray')
ax3.set_title('Matched')

plt.tight_layout()
plt.show()
