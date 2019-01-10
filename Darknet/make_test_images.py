#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make IE613 Test images for Darknet.


"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import time
import os
import pdb
from matplotlib import dates
from radiospectra import spectrogram
from datetime import datetime
from astropy.convolution import convolve, Tophat2DKernel
from skimage.transform import resize
from matplotlib import gridspec
from spectro_prep import rfi_removal, backsub



def write_png_for_classifier(data, filename):

    #-------------------------------------------------------#
    #
    #    Resize, remove rfi, background subtract data
    #
    #tophat_kernel = Tophat2DKernel(3)
    #data_smooth = convolve(data, tophat_kernel)
    data = data[::-1, ::]
    #data = rfi_removal(data, boxsz=1)
    data = backsub(data)
    scl0 = 0.995 #data.mean() #+ data.std()     
    scl1 = 1.025 #data.mean() + data.std()*4.0
    # Note these intensity scaling factors are important. If the background is not clipped away, 
    # the classifier is not successful. Only the most intense bursts in the image are classified. 
    # This is because the training data had to be clipped quite harshly to train the CNN.

    #-------------------------------------------------------------------#
    #
    #    Write png that will be ingested by Tensorflow trained model
    #    
    #data[::]=1.0
    fig = plt.figure(1, frameon=False, figsize=(4,4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(data, cmap=plt.get_cmap('gray'), vmin=scl0, vmax=scl1)
    fig.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

    #img1 = path+'/training_'+str(format(iamgenum, '04'))+'.png'
    #fig.savefig(img1, transparent = True, bbox_inches = 'tight', pad_inches = 0)


IE613_file = '20170910_070804_bst_00X.npy'
event_date = IE613_file.split('_')[0]
file_path = '../Inception/classify_'+event_date+'/'
output_path = 'data/IE613_test/'
input_height = 512
input_width = 512
timestep = int(30.0)   # Seconds 

#-------------------------------------#
#
#      Read in IE613 spectrogram
#
result = np.load(file_path+IE613_file)
spectro = result[0]['data']                     # Spectrogram of entire day
freqs = np.array(result[0]['freq'])             # In MHz
timesut_total = np.array(result[0]['time'])     # In UTC

# Sort frequencies
spectro = spectro[::-1, ::]                     # Reverse spectrogram. For plotting high -> low frequency
freqs = freqs[::-1]                             # For plotting high -> low frequency
#spectro = spectro_process.backsub(spectro)
indices = np.where( (freqs>=20.0) & (freqs<=100.0) )              # Taking only the LBA frequencies
freqs = freqs[indices[0]]
spectro = spectro[indices[0], ::]

# Sort time
block_sz = 10.0 # minutes
time_start = timesut_total[0] #datetime(2017, 9, 2, 10, 46, 0).timestamp() 
time0global = time_start 
time1global = time_start  + 60.0*block_sz      
deltglobal = timesut_total[-1] - timesut_total[0]
trange = np.arange(0, deltglobal, timestep)


for img_index, tstep in enumerate(trange):

    #-------------------------------------#
    #     Select block of time. 
    #     Shifts by tstep every iteration of the loop.
    #
    time_start = time0global + tstep
    time_stop = time1global + tstep
    time_index = np.where( (timesut_total >= time_start) 
                            & (timesut_total <= time_stop))
    times_ut = timesut_total[time_index[0]]
    data = spectro[::, time_index[0]] #mode 3 reading
    delta_t = times_ut - times_ut[0]
    times_dt = [datetime.fromtimestamp(t) for t in times_ut]
    time0_str = times_dt[0].strftime('%Y%m%d_%H%M%S')
    data_resize = resize(data, (input_height, input_height))
    png_file = output_path+'/image_'+time0_str+'.png'
    print('Writing %s' %(png_file))
    write_png_for_classifier(data_resize, png_file) 
    os.system("mogrify -format jpg -trim -resize 512x512 "+png_file)
    #pdb.set_trace()
    
#ffmpeg -y -r 25 -i image_%04d.png -vb 50M classified.mpg
