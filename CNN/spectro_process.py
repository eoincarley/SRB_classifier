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
        box = np.clip([yind-boxsz*2, yind+boxsz*2, xind-boxsz, xind+boxsz], 0, np.shape(image)[0]-1)
        section = image[  box[0]:box[1], box[2]:box[3] ]
        image[box[0]:box[1], box[2]:box[3]] = np.median(section)

    return image     

def backsub(data):
    # Devide each spectrum by the spectrum with the minimum standard deviation.
    data = np.log10(data)
    data[np.where(np.isinf(data)==True)] = 0.0
    data_std = np.std(data, axis=0)
    min_std_index = np.where(data_std==np.min( data_std[np.nonzero(data_std)] ))[0][0]
    min_std_spec = data[:, min_std_index]
    nfreq = len(min_std_spec)
    data = np.divide(data, min_std_spec.reshape(nfreq, 1))
    #Alternative: Normalizing frequency channel responses using median of values.
    #for sb in np.arange(data.shape[0]):
    #       data[sb, :] = data[sb, :]/np.mean(data[sb, :])
    return data    