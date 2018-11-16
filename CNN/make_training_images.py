#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#Type II analysis - band splitting

27March - fitting gaussian animation - best to date

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import time
import os
import pdb
import seaborn as sns
from matplotlib import dates
from radiospectra import spectrogram
from datetime import datetime
from astropy.convolution import convolve, Tophat2DKernel
from skimage.transform import resize
from matplotlib import gridspec

def add_label(taxis, prob, bursttype):
    ax1.text(taxis[-1], prob[-1], bursttype+' ('+str(round(prob[-1],2))+')', fontsize=8)


def backsub(data):
	# Devide each spectrum by the spectrum with the minimum standard deviation.
    data = np.log10(data)
    data[np.where(np.isinf(data)==True)] = 0.0
    data_std = np.std(data, axis=0)
    min_std_index=np.where(data_std==np.min( data_std[np.nonzero(data_std)] ))[0][0]
    min_std_spec = data[:, min_std_index]
    data=np.transpose(np.divide(np.transpose(data), min_std_spec ))

    #Alternative: Normalizing frequency channel responses using median of values.
	#for sb in np.arange(data.shape[0]):
	#   	data[sb, :] = data[sb, :]/np.mean(data[sb, :])
    return data


################################
#         Read data
#
path = './'
file = '20170902_bst_00X_lv1.npy'
result=np.load(path+file)
datatotal = result[0]['data']
datatotal= backsub(datatotal)
datatotal = datatotal[::-1, ::]
timesut_total = np.array(result[0]['time'])  # In UTC
freqs = np.array(result[0]['freq'])  # In MHz
freqs = freqs[::-1]
indices = np.where(freqs <=100.0)
freqs = freqs[indices[0]]
datatotal = datatotal[indices[0], ::]
iamgenum=0
timestep=60.0

for delt in np.arange(0, 28800, timestep):
    ################################
    #         Sort data
    #
    time_start = datetime(2017, 9, 2, 10, 38).timestamp() + 1.0*delt
    time_stop = datetime(2017, 9, 2, 10, 43).timestamp() + 1.0*delt
    time_index = np.where( (timesut_total >= time_start) & (timesut_total <= time_stop))
    times_ut = timesut_total[time_index[0]]
    data = datatotal[::, time_index[0]] #mode 3 reading
    
    delta_t = times_ut - times_ut[0]
    times_dt = [datetime.fromtimestamp(t) for t in times_ut]

    ################################
    #    Smooth and resize data
    #
    tophat_kernel = Tophat2DKernel(3)
    data_smooth = convolve(data, tophat_kernel)
    data_resize = resize(data, (300, 300))
    data_resize = data_resize[::-1, ::]

    #####################
    #    Write png  
    #    
    fig = plt.figure(1, frameon=False, figsize=(4,4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(data_resize, cmap=plt.get_cmap('gray'), vmin=0.96, vmax=data_resize.max())
    img1 = path+'training_imgs/training_'+str(format(iamgenum, '04'))+'.png'    
    fig.savefig(img1, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)
    iamgenum += 1


