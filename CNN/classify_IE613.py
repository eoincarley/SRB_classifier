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
data = result[0]['data']
data = backsub(data)
times_ut = np.array(result[0]['time'])  # In UTC
freqs = np.array(result[0]['freq'])  # In MHz

################################
#         Sort data
#
time_start = datetime(2017, 9, 2, 10, 48).timestamp()
time_stop = datetime(2017, 9, 2, 10, 58).timestamp()
time_index = np.where( (times_ut >= time_start) & (times_ut <= time_stop))
times_ut = times_ut[time_index[0]]
data = data[::, time_index[0]] #mode 3 reading

freqs = freqs[::-1]
data = data[::-1, ::]
indices = np.where(freqs <=100.0)
freqs = freqs[indices[0]]
data = data[indices[0], ::]
delta_t = times_ut - times_ut[0]
times_dt = [datetime.fromtimestamp(t) for t in times_ut]

################################
#    Smooth and resize data
#
tophat_kernel = Tophat2DKernel(2)
data_smooth = convolve(data, tophat_kernel)
data_resize = resize(data_smooth, (100,100))
data_resize = data_resize[::-1, ::]

###########################################################
#    Write png and execute Tensorflow label script 
#   
fig = plt.figure(1, frameon=False, figsize=(4,4))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(data_resize, cmap=plt.get_cmap('gray'), vmin=0.95, vmax=1.2)
fig.savefig(path+'classify_20170902/input.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.close()
os.system('./label_image.sh')

##########################################
#    Plot unsmooth dynamic spectrum 
#   
fig = plt.figure(2, figsize=(10,7))
ax0 = fig.add_axes([0.1, 0.11, 0.9, 0.6])
spec=spectrogram.Spectrogram(data, delta_t, freqs, times_dt[0], times_dt[-1])
spec.t_label='Time (UT)'
spec.f_label='Frequency (MHz)'
spec.plot(vmin=0.95, vmax=1.4, cmap=plt.get_cmap('Spectral_r'))


##########################################
#       Read label results 
# 
file = open(path+'classify_20170902/class_probs.txt', 'r')
labels = file.read()
labels = [txt.split(' ') for txt in labels.split('\n')] 
labels = labels[0:-1]
type0prob = [float(txt[1]) for txt in labels if txt[0]=='type0']*len(delta_t)
typeIIprob = [float(txt[1]) for txt in labels if txt[0]=='typeii']*len(delta_t)
typeIIIprob = [float(txt[1]) for txt in labels if txt[0]=='typeiii']*len(delta_t)
complexprob = [float(txt[1]) for txt in labels if txt[0]=='complex']*len(delta_t)

sns.set()
sns.set_style("darkgrid")
ax1 = fig.add_axes([0.1, 0.72, 0.72, 0.25])
plt.plot(delta_t, type0prob)
plt.plot(delta_t, typeIIprob)
plt.plot(delta_t, typeIIIprob)
plt.plot(delta_t, complexprob)

add_label(delta_t, type0prob, 'No burst')
add_label(delta_t, typeIIprob, 'Type II')
add_label(delta_t, typeIIIprob, 'Type III')
add_label(delta_t, complexprob, 'Complex')

ax1.set_ylim([0,1])
plt.autoscale(enable=True, axis='x', tight=True)
#plt.xticks([])
ax1.set_xticklabels([' '])
plt.ylabel('Detection Probability')
ax1.yaxis.label.set_size(10)
plt.show()
#plt.close()
sns.reset_orig()












