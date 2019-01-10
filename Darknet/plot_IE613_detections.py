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


#-------------------------------------#
#
#      		 Functions
#
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


def get_box_coords(coords, nx, ny):

	# Awkardly, the x0, y0 points for the patches.rectangle procedure is a little 
	# different to how the coords are defined by yolo. This converts from one to the other.

	x0, y0 = int(coords[0]*nx), int(coords[1]*ny)
	y0plot = coords[1]*ny    #the spec plotter seems to double the number of y-axis pixels.

	width, height = coords[2]*nx, coords[3]*ny
	y1 = ny - (y0plot + height)  # Note that pacthes.Rectangle measure the y-coord from the bottom.
	x1 = int(x0 + width)

	spill=x1-nx-2.0 if x1>=(nx-2) else 0.0
	width = width-spill

	boxx0 = np.clip(x0, 0, nx)
	boxy0 = np.clip(y1, 0, ny)

	return x0, y0, x1, y1, boxx0, boxy0, width, height



#-------------------------------------#
#
#      		 Main procedure
#
IE613_file = '20170910_070804_bst_00X.npy'
event_date = IE613_file.split('_')[0]
file_path = '../Inception/classify_'+event_date+'/'
output_path = 'data/IE613_predictions/'
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
#spectro = backsub(spectro)
indices = np.where( (freqs>=20.0) & (freqs<=100.0) )              # Taking only the LBA frequencies
freqs = freqs[indices[0]]
spectro = spectro[indices[0], ::]
nfreqs = len(freqs)
nspecfreqs = nfreqs*2 # the spectro plotter doubles the amount of y points.

# Sort time
block_sz = 10.0 # minutes
time_start = timesut_total[0] #datetime(2017, 9, 2, 10, 46, 0).timestamp() 
time0global = time_start 
time1global = time_start  + 60.0*block_sz      
deltglobal = timesut_total[-1] - timesut_total[0]
trange = np.arange(0, deltglobal, timestep)

yolo_allburst_coords = yolo_results_parser('IE613_detections_600_0010_20190110.txt')

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
	yolo_burst_coords = yolo_allburst_coords[img_key]


	fig = plt.figure(2, figsize=(10,7))
	ax = fig.add_axes([0.1, 0.11, 0.9, 0.8])
	spec=spectrogram.Spectrogram(data, delta_t, freqs, times_dt[0], times_dt[-1])
	spec.t_label='Time (UT)'
	spec.f_label='Frequency (MHz)'
	scl1 = 0.995 #data.mean() #- data.std()
	scl2 = 1.025 #data.mean() + data.std()*4.0
	spec.plot(vmin=scl1, 
	          vmax=scl2, 
	          cmap=plt.get_cmap('bone'))

	ntimes = len(delta_t)
	npoints = 0
	for burstcoords in yolo_burst_coords:
		burstcoords = np.array(burstcoords)
		burstcoords = np.clip(burstcoords, 4, 508)/input_width

		x0, y0, x1, y1, boxx0, boxy0, width, height = get_box_coords(burstcoords, ntimes, nspecfreqs)

		rect = patches.Rectangle( (boxx0, boxy0), width, height, 
				linewidth=0.5, 	
				edgecolor='lawngreen', 
				facecolor='none')  
		ax.add_patch(rect)

		#----------------------------------#
		#
		#	Plot red points inside boxes
		#	
		y1 = int(y0 + height/2.0)
		y0index = nfreqs - y1 #nfreqs - y0+height/2
		y1index = nfreqs - y0

		data_section = data[y0index:y1index, x0:x1]
		thresh = np.median(data_section)+data_section.std()*0.7
		burst_indices = np.where(data>thresh)

		xpoints, ypoints = burst_indices[1], burst_indices[0]
		box_indices = (xpoints>x0) & \
					  (xpoints<x1) & \
					  (ypoints<y1index) & \
					  (ypoints>y0index)

		xbox = xpoints[np.where( box_indices )]
		ybox = ypoints[np.where( box_indices )]
		xbox, ybox = np.clip(xbox, 0, ntimes-2), np.clip(ybox, 0, nfreqs-2)
		npoints = npoints + len(xpoints)

		plt.scatter(x=xbox, y=ybox*2.0, c='r', s=10, alpha=0.1)
	

	plt.text(200, 365, 'IE613 I-LOFAR YOLOv3 type III detections')
	out_png = output_path+'/IE613_'+str(format(img_index, '04'))+'_detections.png'
	print("Saving %s" %(out_png))
	fig.savefig(output_path+'/IE613_'+str(format(img_index, '04'))+'_detections.png')
	#plt.show()
	#pdb.set_trace()
	plt.close(fig)
	#if img_index==19: pdb.set_trace()



    
#ffmpeg -y -r 20 -i IE613_%04d_detections.png -vb 50M IE613_YOLO_600_0001.mpg
