# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


def mse(id, od):
    return (np.square(id.astype(int) - od.astype(int))).mean(axis=None)


def Local_median(img_o, f_size):
    median = cv2.medianBlur(img_o, f_size)
    img_o = np.where((img_o == 0) | (img_o == 255), median, img_o)
    return img_o


def Local_Gaussian(img_n, img_o, k_size, sigma):
    gauss = cv2.GaussianBlur(img_o, (k_size, k_size), sigma)
    img_o = np.where((img_n == 0) | (img_n == 255), gauss, img_o)
    return img_o


img = r"rdata4.bmp"
img_n = r"idata4.bmp"

img_i = cv2.imread(img)     # Original Image
img_o = cv2.imread(img_n)   # Output Image
img_n = cv2.imread(img_n)   # Noise Image
img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
img_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)

# Filtering process
# Median Filter
F_SIZE = 3  # n*n filter
immse = 1
k = 0   # Number of attempts

while(immse != 0):
    img_b = img_o   # Image before filtering process
    img_o = Local_median(img_o, F_SIZE)

    immse = mse(img_b, img_o)
    k += 1

print('Number of attempts (median):' + str(k))

# Gaussian Filter
SIGMA = 0.55    # Standard deviation
K_SIZE = 5  # kernelsize
immse = 1
k = 0

while(immse != 0):
    img_b = img_o
    img_o = Local_Gaussian(img_n, img_o, K_SIZE, SIGMA)

    immse = mse(img_b, img_o)
    k += 1

print('Number of attempts (Gaussian):' + str(k))
print('Result of mse:{:.2f}'.format(mse(img_i, img_o)))

# Display results
# plt.figure('Execution result')
# plt.subplot(131), plt.imshow(img_i, cmap='gray'), plt.title('Original Image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(img_n, cmap='gray'), plt.title('Noise Image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(img_o, cmap='gray'), plt.title('Filterd Image')
# plt.xticks([]), plt.yticks([])
# plt.show()
