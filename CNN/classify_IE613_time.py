#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convolutional neural network radio burst classifier.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import time
import os
import pdb
import seaborn as sns
import label_image
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

def tf_label(graph, label_file):
    t = label_image.read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
    })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = label_image.load_labels(label_file)

    results = list(zip(labels, results))
    print(results)
    return results
    

################################
#         Read data
#

event_date = '20170902'
file_path = './classify_'+event_date+'/'
output_path = './classify_'+event_date+'/trial2/'
file = file_path+'20170902_103626_bst_00X.npy'
result = np.load(file)
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

model_file = '/tmp/output_graph.pb'
file_name = output_path+'/input.png'
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"
label_file = "/tmp/output_labels.txt"

graph = label_image.load_graph(model_file)
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

time0global = datetime(2017, 9, 2, 10, 38).timestamp()
time1global = datetime(2017, 9, 2, 10, 43).timestamp() 
deltglobal = time1global - time0global
timestep=10

for delt in np.arange(0, deltglobal, timestep):
    ################################
    #         Sort data
    #

    time_start = time0global + delt
    time_stop = time1global + delt
    time_index = np.where( (timesut_total >= time_start) & (timesut_total <= time_stop))
    times_ut = timesut_total[time_index[0]]
    data = datatotal[::, time_index[0]] #mode 3 reading
    
    delta_t = times_ut - times_ut[0]
    times_dt = [datetime.fromtimestamp(t) for t in times_ut]


    ################################
    #    Smooth and resize data
    #
    #tophat_kernel = Tophat2DKernel(3)
    #data_smooth = convolve(data, tophat_kernel)
    data_resize = resize(data, (300, 300))
    data_resize = data_resize[::-1, ::]


    ###########################################################
    #    Write png and execute Tensorflow label script 
    #    
    fig = plt.figure(1, frameon=False, figsize=(4,4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    scl0 = data_resize.max()*0.75     #data_resize.mean() + data_resize.std()       # was data_resize.max()*0.75
    scl1 = data_resize.max()*0.9      #data_resize.mean() + data_resize.std()*4.0   # data_resize.max()
    ax.imshow(data_resize, cmap=plt.get_cmap('gray'), vmin=scl0, vmax=scl1)
    fig.savefig(file_name, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

    #img1 = path+'/training_'+str(format(iamgenum, '04'))+'.png'
    #fig.savefig(img1, transparent = True, bbox_inches = 'tight', pad_inches = 0)

    
    #####################################################
    #
    #    Execute Tensorflow label function
    # 
    burst_probs = tf_label(graph, label_file)
    type0prob = np.array( [burst_probs[0][1]] )   
    typeIIprob = np.array( [burst_probs[1][1]] )   
    typeIIIprob = np.array( [burst_probs[2][1]] )  

    if delt==0:
        timprobs = delta_t[0:-1:timestep]
        type0probt = type0prob.repeat(len(timprobs))
        typeIIprobt = typeIIprob.repeat(len(timprobs))
        typeIIIprobt = typeIIIprob.repeat(len(timprobs))
    else:
        type0probt = np.concatenate( (type0probt[1::], type0prob) )  
        typeIIprobt = np.concatenate( (typeIIprobt[1::], typeIIprob) )  
        typeIIIprobt = np.concatenate( (typeIIIprobt[1::], typeIIIprob) )  

    ##########################################
    #    Plot unsmooth dynamic spectrum 
    #   
    fig = plt.figure(2, figsize=(10,7))
    ax0 = fig.add_axes([0.1, 0.11, 0.9, 0.6])
    spec=spectrogram.Spectrogram(data, delta_t, freqs, times_dt[0], times_dt[-1])
    spec.t_label='Time (UT)'
    spec.f_label='Frequency (MHz)'
    spec.plot(vmin=0.96, vmax=1.3, cmap=plt.get_cmap('Spectral_r'))

    #sns.set()
    #sns.set_style("darkgrid")
    ax1 = fig.add_axes([0.1, 0.72, 0.72, 0.25])
    plt.plot(timprobs, type0probt, color='blue')
    plt.plot(timprobs, typeIIprobt, color='red')
    plt.plot(timprobs, typeIIIprobt, color='green')

    add_label(timprobs, type0probt, 'No burst')
    add_label(timprobs, typeIIprobt, 'Type II')
    add_label(timprobs, typeIIIprobt, 'Type III')

    ax1.set_ylim([0, 1])
    ax1.autoscale(enable=True, axis='x', tight=True)
    plt.xticks([])
    ax1.set_xticklabels([' '])
    plt.ylabel('Detection Probability')
    ax1.yaxis.label.set_size(10)


    fig.savefig(output_path+'/image_'+str(format(iamgenum, '04'))+'.png')
    iamgenum += 1
    plt.close(fig)

#ffmpeg -y -r 25 -i image_%04d.png -vb 50M classified.mpg
