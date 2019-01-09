#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot the YOLOv3 detections on the dynamic spectra


"""
import os
import pdb
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import matplotlib.patches as patches
from matplotlib import dates
from radiospectra import spectrogram
from datetime import datetime
from astropy.convolution import convolve, Tophat2DKernel
from skimage.transform import resize
from matplotlib import gridspec
from spectro_prep import rfi_removal, backsub 

def yolo_results_parser(txt):
    file = open(txt, 'r')
    yolo_results = np.array(list(file))

    yolo_dict = {}
    img_indices = []
    [img_indices.append(index) for index, row in enumerate(yolo_results) if row[0]=='E']

    for i in np.arange(len(img_indices)-1):
        bursts = []
        result = yolo_results[img_indices[i]:img_indices[i+1]]
        img_name = result[0].split('/')[2].split('.')[0]
        print(img_name)
        txt_coords = result[1::]
        for txt_coord in txt_coords:
            coords = np.array(txt_coord.split('(')[1].split(')')[0].split(' '))
            coords = coords[coords!='']
            bursts.append(coords[[1,3,5,7]].astype(np.float).tolist())
        yolo_dict[img_name]=bursts

    return yolo_dict    


IE613_file = '20170910_070804_bst_00X.npy'
event_date = IE613_file.split('_')[0]
file_path = '../Inception/classify_'+event_date+'/'
output_path = 'data/IE613_predictions/'
input_height = 512
input_width = 512
timestep = int(10.0)   # Seconds 

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
#spectro = backsub(spectro)
indices = np.where( (freqs>=20.0) & (freqs<=100.0) )              # Taking only the LBA frequencies
freqs = freqs[indices[0]]
spectro = spectro[indices[0], ::]
nfreqs = len(freqs)

# Sort time
block_sz = 5.0 # minutes
time_start = timesut_total[0] #datetime(2017, 9, 2, 10, 46, 0).timestamp() 
time0global = time_start 
time1global = time_start  + 60.0*block_sz      
deltglobal = timesut_total[-1] - timesut_total[0]
trange = np.arange(0, deltglobal, timestep)

yolo_burst_coords = yolo_results_parser('IE613_detections_600_0005_20190108.txt')

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
    data = backsub(data)
    delta_t = times_ut - times_ut[0]
    times_dt = [datetime.fromtimestamp(t) for t in times_ut]
    time0_str = times_dt[0].strftime('%Y%m%d_%H%M%S')
    img_key = 'image_'+time0_str
    burst_coords = yolo_burst_coords[img_key]


    fig = plt.figure(2, figsize=(10,7))
    ax = fig.add_axes([0.1, 0.11, 0.9, 0.8])
    spec=spectrogram.Spectrogram(data, delta_t, freqs, times_dt[0], times_dt[-1])
    spec.t_label='Time (UT)'
    spec.f_label='Frequency (MHz)'
    spec.plot(vmin=data.mean() - data.std(), 
              vmax=data.mean() + data.std()*4.0, 
              cmap=plt.get_cmap('bone'))

    ntimes = len(delta_t)
    npoints = 0
    for burst in burst_coords:
        burst = np.array(burst)
        burst = np.clip(burst, 5, 505)/512
        x0 = burst[0]*ntimes
        y0 = nfreqs - burst[1]*nfreqs
        y0plot = nfreqs*2.0 - burst[1]*nfreqs*2.0    #the spec plotter seems to double the number of y-axis pixels.

        width = burst[2]*ntimes
        height = burst[3]*nfreqs*2.0
        y1 = (y0plot - height)
        x1 = x0 + width
        if x1>ntimes: 
            spill = x1-ntimes
            width = width-spill - 5.0

        rect = patches.Rectangle((x0, np.clip(y1, 3, 500)), width, height, linewidth=0.5, edgecolor='lawngreen', facecolor='none')  
        ax.add_patch(rect)

        x0index = burst[0]*ntimes
        x1index = x0index+width

        y1index = nfreqs-burst[1]*nfreqs
        y0index = y1index - height/2

        thresh = data.mean()+data.std()*0.7
        burst_indices = np.where(data>thresh)

        xpoints = burst_indices[1]
        ypoints = burst_indices[0]
        xbox = xpoints[np.where( (xpoints>x0index) & (xpoints<x1index) & (ypoints<y1index) & (ypoints>y0index) )]
        ybox = ypoints[np.where( (xpoints>x0index) & (xpoints<x1index) & (ypoints<y1index) & (ypoints>y0index) )]
        xbox = np.clip(xbox, 0, ntimes-3)
        ybox = np.clip(ybox, 0, nfreqs-2)
        npoints = npoints + len(xpoints)
        #xpoints = xpoints[np.where(xpoints>x0index and xpoints<x1index)]
        #ypoints = ypoints[np.where(xpoints>x0index and xpoints<x1index)]

        plt.scatter(x=xbox, y=ybox*2.0, c='r', s=10, alpha=0.1)


    plt.text(100, 375, 'IE613 I-LOFAR YOLOv3 type III detections')
    out_png = output_path+'/IE613_'+str(format(img_index, '04'))+'_detections.png'
    print("Saving %s" %(out_png))
    fig.savefig(output_path+'/IE613_'+str(format(img_index, '04'))+'_detections.png')
    #plt.show()
    #pdb.set_trace()
    plt.close(fig)
    
    
#ffmpeg -y -r 25 -i IE613_%04d_detections.png -vb 50M IE613_YOLO.mpg
