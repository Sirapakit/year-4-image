# Contrast stretching

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = r'../low-contrast.jpg'
image_cv2 = cv2.imread(path)


# Method to process the red band of the image
def normalize_red(intensity):
    intensity_value = intensity
    min_intensity = 86
    max_intensity = 230
    min_o = 0
    max_o = 255

    return (intensity_value-min_intensity)*(((max_o-min_o)/(max_intensity-min_intensity))+min_o)

# Method to process the green band of the image


def normalized_green(intensity):
    intensity_value = intensity
    min_intensity = 90
    max_intensity = 225
    min_o = 0
    max_o = 255

    return (intensity_value-min_intensity)*(((max_o-min_o)/(max_intensity-min_intensity))+min_o)

# Method to process the blue band of the image


def normalize_blue(intensity):
    intensity_value = intensity
    min_intensity = 100
    max_intensity = 210
    min_o = 0
    max_o = 255
    return (intensity_value-min_intensity)*(((max_o-min_o)/(max_intensity-min_intensity))+min_o)


# Create an image object
image_object = Image.open("../low-contrast.jpg")

# Split the red, green and blue bands from the Image
multibands = image_object.split()

# Apply point operations that does contrast stretching on each color band
normalized_red_band = multibands[0].point(normalize_red)
normalized_green_band = multibands[1].point(normalized_green)
normalized_blue_band = multibands[2].point(normalize_blue)

# Create a new image from the contrast stretched red, green and blue brands
normalized_image = Image.merge(
    "RGB", (normalized_red_band, normalized_green_band, normalized_blue_band))

# Display the image before contrast stretching
image_object.show()

# Display the image after contrast stretching
normalized_image.show()
normalizedImageArray = np.asarray(normalized_image)  # a is readonly

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot for original image
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image_cv2[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

# create the histogram plot for constrast image
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        normalizedImageArray[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

plt.show()
