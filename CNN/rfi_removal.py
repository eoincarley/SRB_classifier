#!/usr/bin/env python3

"""
 File:
    rfi_removal.py

 Description:
    Find where the sobel gradient is high and replace those pixels with the median
    of their surroundings. Because RFI tends to be narrow band the vertical surroudnings
    are larger than the horizontal

 Version hitsory:

	Created 2018-May-05. E. Carley

"""

import numpy as np
from scipy import ndimage

def rfi_removal(image, boxsz=5):
    sx = ndimage.sobel(image, axis=0, mode='constant')
    sy = ndimage.sobel(image, axis=1, mode='constant')
    data_sobel = np.abs(np.hypot(sx, sy))
    thold = data_sobel.mean() + data_sobel.std()*1.0
    indices = np.where(data_sobel>thold)
    for index, value in enumerate(indices[0]):
        xind = indices[1][index]
        yind = indices[0][index]
        box = np.clip([yind-boxsz, yind+boxsz*2, xind-boxsz*2, xind+boxsz], 0, np.shape(image)[0]-1)
        section = image[  box[0]:box[1], box[2]:box[3] ]
        image[box[0]:box[1], box[2]:box[3]] = np.median(section)

    return image     