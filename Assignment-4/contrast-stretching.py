# Contrast stretching

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = r'../low-contrast.jpg'
imageCv2 = cv2.imread(path)


# Method to process the red band of the image
def normalizeRed(intensity):
    iI = intensity
    minI = 86
    maxI = 230
    minO = 0
    maxO = 255
    return (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

# Method to process the green band of the image


def normalizeGreen(intensity):
    iI = intensity
    minI = 90
    maxI = 225
    minO = 0
    maxO = 255
    return (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

# Method to process the blue band of the image


def normalizeBlue(intensity):
    iI = intensity
    minI = 100
    maxI = 210
    minO = 0
    maxO = 255
    return (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)


# Create an image object
imageObject = Image.open("../low-contrast.jpg")

# Split the red, green and blue bands from the Image
multiBands = imageObject.split()

# Apply point operations that does contrast stretching on each color band
normalizedRedBand = multiBands[0].point(normalizeRed)
normalizedGreenBand = multiBands[1].point(normalizeGreen)
normalizedBlueBand = multiBands[2].point(normalizeBlue)

# Create a new image from the contrast stretched red, green and blue brands
normalizedImage = Image.merge(
    "RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

# Display the image before contrast stretching
imageObject.show()

# Display the image after contrast stretching
normalizedImage.show()
normalizedImageArray = np.asarray(normalizedImage)  # a is readonly

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot for original image
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        imageCv2[:, :, channel_id], bins=256, range=(0, 256)
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
