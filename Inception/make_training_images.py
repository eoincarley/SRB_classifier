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
from spectro_prep import backsub, rfi_removal
from matplotlib import dates
from radiospectra import spectrogram
from datetime import datetime
from astropy.convolution import convolve, Tophat2DKernel
from skimage.transform import resize
from matplotlib import gridspec
from scipy import ndimage

def add_label(taxis, prob, bursttype):
    ax1.text(taxis[-1], prob[-1], bursttype+' ('+str(round(prob[-1],2))+')', fontsize=8)


################################
#         Read data
#
path = 'classify_20170902/'
file = '20170902_103626_bst_00X.npy'
result=np.load(path+file)
datatotal = result[0]['data']

datatotal = datatotal[::-1, ::]
timesut_total = np.array(result[0]['time'])  # In UTC
freqs = np.array(result[0]['freq'])  # In MHz
freqs = freqs[::-1]
indices = np.where(freqs <=100.0)
freqs = freqs[indices[0]]
datatotal = datatotal[indices[0], ::]
iamgenum=0
timestep=60.0
downsize=300

for delt in np.arange(0, 28800, timestep):
    ################################
    #         Sort data
    #
    time_start = datetime(2017, 9, 2, 10, 38).timestamp() + 1.0*delt
    time_stop = datetime(2017, 9, 2, 10, 48).timestamp() + 1.0*delt
    time_index = np.where( (timesut_total >= time_start) & (timesut_total <= time_stop))
    times_ut = timesut_total[time_index[0]]
    data = datatotal[::, time_index[0]] #mode 3 reading
    datatotal=[0]
    
    delta_t = times_ut - times_ut[0]
    times_dt = [datetime.fromtimestamp(t) for t in times_ut]

    ###################################################
    #    Resize, RFI removal, background subtract
    #
    data_resize = resize(data, (downsize, downsize))
    data_resize = data_resize[::-1, ::]
    data_resize = rfi_removal(data_resize, boxsz=1)
    data_resize = backsub_min(data_resize)


    ################################
    #    Smooth and resize data
    #
    #tophat_kernel = Tophat2DKernel(3)
    #data_resize = convolve(data_resize, tophat_kernel)
    scl0 = data_resize.max()*0.8
    scl1 = data_resize.max()*0.9
 
    #####################
    #    Write png  
    #    
    fig = plt.figure(1, frameon=False, figsize=(4,4))
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.axis('off')
    plt.imshow(data_resize, cmap=plt.get_cmap('gray'), vmin=scl0, vmax=scl1)
    #plt.scatter(indices[1], indices[0], c='r', s=1)
    #img1 = path+'training_imgs/training_'+str(format(iamgenum, '04'))+'.png'    
    #fig.savefig(img1, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    #plt.close(fig)
    plt.show()
    pdb.set_trace()
    iamgenum += 1


