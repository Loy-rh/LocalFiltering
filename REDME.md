# LocalFiltering
4th year seminar assignment

High-performance noise reduction algorithm

## Local Filtering Algorithm
1.Identify the noise pixels and filter only those areas.<br>
2.Apply a median filter to a specific area to remove noise pixel values.<br>
3.Apply a Gaussian filter to pixels that <br>
    originally had noise to smooth out the color changes.

## Features
It can remove noise and restore the original image with higher accuracy than just filtering.

## Dependent Libraries

numpy 1.19.2<br>
opencv 4.0.1<br>
matplotlib 3.3.2

## Installation
* opencv<br>
  * anaconda3: conda install opencv
  * pip: pip install opencv-python

# Usage
img: Path of the original image with no noise added<br>
img_n: Path of the image with added noise.<br>

## Note
This source code can only be applied to noise with luminance values of 0 or 255.

## Author

Ryota Higashimoto <br>
Image Processing Engineering Laboratory <br>
