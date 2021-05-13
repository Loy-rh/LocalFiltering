# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


def mse(id, od):
    return (np.square(id.astype(int) - od.astype(int))).mean(axis=None)


img = r"rdata4.bmp"
img_n = r"idata4.bmp"

img_i = cv2.imread(img)     # Original Image
img_o = cv2.imread(img_n)   # Output Image
img_n = cv2.imread(img_n)   # Noise Image
img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
img_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)

# Histogram
# hist_i = cv2.calcHist([img_i], [0], None, [256], [0, 256])
# hist_o = cv2.calcHist([img_o], [0], None, [256], [0, 256])
# plt.figure('Histogram')
# plt.plot(hist_o, label='Noise')
# plt.plot(hist_i, label='Original')
# plt.plot(hist_o - hist_i, label='Noise-Original')
# plt.xlim([0, 256])
# plt.legend()
# plt.show()

# Filtering process
# Median Filter
f_size = 5  # n*n filter
median = cv2.medianBlur(img_o, f_size)
ol1, ol2 = np.where((img_o == 0) | (img_o == 255))
for i, j in zip(ol1, ol2):
    img_o[i, j] = median[i, j]

mse_b = mse(img_i, median)  # before
mse_a = mse(img_i, img_o)   # after
# print('median:' + str(mse_b))
# print('1：' + str(mse_a))

k = 2
while(mse_a < mse_b):
    mse_b = mse_a
    median = cv2.medianBlur(img_o, f_size)
    l1, l2 = np.where((img_o == 0) | (img_o == 255))
    for i, j in zip(l1, l2):
        img_o[i, j] = median[i, j]

    mse_a = mse(img_i, img_o)
    # print(str(k) + '：' + str(mse_a))
    k += 1

# Gaussian Filter
sigma = 0.55    # Standard deviation
k_size = 5  # kernelsize
gauss = cv2.GaussianBlur(img_o, (k_size, k_size), sigma)
mse_b = mse(img_i, img_o)
mse_a = mse(img_i, gauss)

while(mse_a < mse_b):
    mse_b = mse_a
    gauss = cv2.GaussianBlur(img_o, (k_size, k_size), sigma)

    for i, j in zip(ol1, ol2):
        img_o[i, j] = gauss[i, j]

    mse_a = mse(img_i, img_o)

print('result:' + str(int(mse_a)))

# Display results
plt.figure('Execution result')
plt.subplot(131), plt.imshow(img_i, cmap='gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_n, cmap='gray'), plt.title('Noise Image')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_o, cmap='gray'), plt.title('Filterd Image')
plt.xticks([]), plt.yticks([])
plt.show()
