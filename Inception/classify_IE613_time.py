#!/usr/bin/env python3

"""
 File:
    classify_IE613_time.py

 Description:
    Classify images of data from I-LOFAR (IE613) to determine what radio burst is in the image.
    Classifier is a trained InceptionV3 model. Tensorflow functions brought over from the retraining
    packages.

 Disclaimer:

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.

 Notes:

 Examples:

 Version hitsory:

    Created 2018-May-05

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf

import os
import time
import pdb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import label_image
import spectro_process
from matplotlib import dates
from radiospectra import spectrogram
from datetime import datetime
from astropy.convolution import convolve, Tophat2DKernel
from skimage.transform import resize
from matplotlib import gridspec


def add_label(taxis, prob, bursttype):
    ax1.text(taxis[-1], prob[-1], bursttype+' ('+str(round(prob[-1],2))+')', fontsize=8)


def tf_label(graph, filename, label_file):
    t = label_image.read_tensor_from_image_file(
    filename,
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
    

def write_png_for_classifier(data, filename):

    #-------------------------------------------------------#
    #
    #    Resize, remove rfi, background subtract data
    #
    #tophat_kernel = Tophat2DKernel(3)
    #data_smooth = convolve(data, tophat_kernel)
    data = data[::-1, ::]
    #data = spectro_process.rfi_removal(data, boxsz=1)
    data = spectro_process.backsub(data)
    scl0 = np.median(data) #data.mean() + data.std()     
    scl1 = np.max(data) #data.mean() + data.std()*4.0
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


IE613_file = '20170902_103626_bst_00X.npy'
event_date = IE613_file.split('_')[0]
file_path = 'classify_'+event_date+'/'
output_path = file_path+'/trial5/'
model_file = '/tmp/output_graph.pb'
png_file = output_path+'/input.png'             # PNG that is output by write_png_for_classifier and ingested by tf_label.
input_height = 299
input_width = 299
input_std = 255
input_mean = 0
input_layer = "Placeholder"
output_layer = "final_result"
input_name = "import/" + input_layer
output_name = "import/" + output_layer
label_file = "/tmp/output_labels.txt"
timestep = int(3.0*60.0)   # Seconds 

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
time_start = timesut_total[0] #datetime(2017, 9, 2, 10, 46, 0).timestamp() 
time0global = time_start 
time1global = time_start  + 60.0*10.0      # +15 minutes
deltglobal = timesut_total[-1] - timesut_total[0]
trange = np.arange(0, deltglobal, timestep)

#-------------------------------------#
#
#     Load in the trained model
#
graph = label_image.load_graph(model_file)
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

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

    data_resize = resize(data, (input_height, input_height))
    write_png_for_classifier(data_resize, png_file) 
    
    #####################################################
    #
    #    Execute Tensorflow label function on PNG file
    # 
    burst_probs = tf_label(graph, png_file, label_file)
    type0prob = np.array( [burst_probs[0][1]] )   
    typeIIprob = np.array( [burst_probs[1][1]] )   
    typeIIIprob = np.array( [burst_probs[2][1]] )  
    #
    ######################################################

    if tstep==0:
        timprobs = delta_t[0:-1:timestep]
        type0probt = type0prob.repeat(len(timprobs))
        typeIIprobt = typeIIprob.repeat(len(timprobs))
        typeIIIprobt = typeIIIprob.repeat(len(timprobs))
    else:
        type0probt = np.concatenate( (type0probt[1::], type0prob) )  
        typeIIprobt = np.concatenate( (typeIIprobt[1::], typeIIprob) )  
        typeIIIprobt = np.concatenate( (typeIIIprobt[1::], typeIIIprob) )  

    #-------------------------------------#
    #
    #    Plot unsmooth dynamic spectrum 
    #   
    fig = plt.figure(2, figsize=(10,7))
    ax0 = fig.add_axes([0.1, 0.11, 0.9, 0.6])
    data = spectro_process.backsub(data)
    #data = spectro_process.rfi_removal(data, boxsz=1)
    spec=spectrogram.Spectrogram(data, delta_t, freqs, times_dt[0], times_dt[-1])
    spec.plot(vmin=np.median(data), 
              vmax=data.max(), 
              cmap=plt.get_cmap('Spectral_r'))
    ax1 = fig.add_axes([0.1, 0.72, 0.72, 0.25])
    plt.ylim((25,100))
    spec.t_label='Time (UT)'
    spec.f_label='Frequency (MHz)'

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

    fig.savefig(output_path+'/image_'+str(format(img_index, '04'))+'.png')
    plt.close(fig)
    pdb.set_trace()
#ffmpeg -y -r 25 -i image_%04d.png -vb 50M classified.mpg
