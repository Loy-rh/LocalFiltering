
# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


def mse(id, od):
    return (np.square(id.astype(int) - od.astype(int))).mean(axis=None)


def Local_Gaussian(img_o, k_size, sigma, l1, l2):
    gauss = cv2.GaussianBlur(img_o, (k_size, k_size), sigma)
    for i, j in zip(l1, l2):
        img_o[i, j] = gauss[i, j]
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
F_SIZE = 5  # n*n filter
median = cv2.medianBlur(img_o, F_SIZE)
img_b = img_o   # Image before filtering process

l1, l2 = np.where((img_o == 0) | (img_o == 255))
for i, j in zip(l1, l2):
    img_o[i, j] = median[i, j]
mse_b = mse(img_b, median)  # before
mse_a = mse(img_b, img_o)   # after
# print('1：' + str(mse_a))

k = 2
while(mse_a != mse_b):
    mse_b = mse_a
    median = cv2.medianBlur(img_o, F_SIZE)
    img_b = img_o
    img_o = np.where((img_o == 0) | (img_o == 255), median, img_o)

    mse_a = mse(img_b, img_o)
    # print(str(k) + '：' + str(mse_a))
    k += 1

print("Number of attempts" + str(k - 1))

# Gaussian Filter
SIGMA = 0.55    # Standard deviation
K_SIZE = 5  # kernelsize
gauss = cv2.GaussianBlur(img_o, (K_SIZE, K_SIZE), SIGMA)
img_o = Local_Gaussian(img_o, K_SIZE, SIGMA, l1, l2)
mse_b = mse(gauss, img_o)
mse_a = mse(gauss, img_b)
# print('1：' + str(mse_a))

k = 2
while(mse_a != mse_b):
    mse_b = mse_a
    img_b = img_o
    img_o = Local_Gaussian(img_o, K_SIZE, SIGMA, l1, l2)

    mse_a = mse(gauss, img_o)
    # print(str(k) + '：' + str(mse_a))
    k += 1

print("Number of attempts" + str(k - 1))
print('result:' + str(int(mse(img_i, img_o))))

# Display results
# plt.figure('Execution result')
# plt.subplot(131), plt.imshow(img_i, cmap='gray'), plt.title('Original Image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(img_n, cmap='gray'), plt.title('Noise Image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(img_o, cmap='gray'), plt.title('Filterd Image')
# plt.xticks([]), plt.yticks([])
# plt.show()
