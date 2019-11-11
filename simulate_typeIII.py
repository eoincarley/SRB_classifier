#!/usr/bin/env python3

"""
 File:
    simulate_typeIII.py

 Description:
   	Simulates a dynamic spectrum of a type III solar radio burst in a dynamic spectrum.
   	Uses random number generaters to vary the drift rate, start-end frequency and times,
   	intensity, bandwidth and a number of other paramaters to ensure a large variety of
   	shapes and sizes.

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
import pdb
import os
from copy import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage.filters as smooth
from PIL import Image
from astropy.convolution import convolve, Tophat2DKernel
from numpy import random


def embed_typeIII(image, trange=[10,400], head_range=[0,200], tail_range=[300,400], intensity=1):

	no_embed=0
	embed_fail=False

	img_sz = np.shape(image)[0]
	t0 = random.randint(trange[0], trange[1])  	# Position of the type III head (top) in time.
	f0 = random.randint(tail_range[0], tail_range[1])    # Position of the type III tail (bottom) in frequency.
	f1 = random.randint(head_range[0], head_range[1])		# Position of the type III head (top) in frequency.
	nfreqs = f0-f1

	times = np.linspace(0, img_sz-1, 1000)
	frange = np.arange(int(f1), int(f0))
	drift_range = np.linspace(1, 5, len(frange))
	base_driftrate = random.randint(0,20)
	xdrift = t0 + base_driftrate/drift_range 		# In pixel units. Responsible for how 'curved' the head is.
	#xdrift = np.linspace(t0, t0-driftrate, len(frange))

	headsize = random.randint(0.1, 15) 	# Controls how much the head of the type III decays away -> how fat the head is.
	tailsize = random.uniform(0.1, 5)     
	turnover = random.randint(3, 100)     # Controls how long the head is in time 						
	tspread0 = np.linspace(0, headsize, turnover)  					# The first ten pixels of the head get fatter
	tspread1 = np.linspace(headsize, tailsize, len(frange[turnover::]))   	# After first ten pixels it get smaller and decays to 0.1
	tspread = np.concatenate((tspread0, tspread1))


	# Controls the intensity along type III body. The tail is randomised in intensity to make it look inhomogeneous
	randpeak = random.randint(3,7)
	peak_flux_at_f = np.repeat(5, len(frange))
	randpeak = random.randint(3,10)
	tail_indices = np.arange( int(len(frange)*0.4), int(len(frange)-1))
	peak_flux_at_f[tail_indices] = random.randint(0, randpeak, len(tail_indices))
	peak_flux_at_f = smooth.gaussian_filter1d(peak_flux_at_f, 6)

	envelope = np.linspace(random.uniform(0.1, 1), 1, len(peak_flux_at_f)) # Sometimes the type III will fade as it increases in frequency.
	peak_flux_at_f = envelope*peak_flux_at_f
	#peak_flux_at_f = random.randint(0, randpeak, len(frange))   # For adding a randomness to the type III intensity along it's body.

	rise_time = 1.0  # Travelling across time for a particular frequency, 
					 # this controls how fast intensity rises. Basically 1 pixel.

	# Following loop controls the rise and decay profile at each frequency.
	fluxrise = 1.0/np.exp(times/1.0) 
	fluxdecay_range = 1/np.linspace(1, 5, len(frange))
	for index, f in enumerate(frange):

		tdecay = tspread[index]  # Decay in time decreases as we go higher in frequency (towards the tail)
		fluxdecay = 1.0/(np.exp(-times/tdecay))
		flux = 1.0/(fluxrise + fluxdecay)

		flux = np.clip(flux*peak_flux_at_f[index], 0, 20)
		flux = flux - flux.min()
		
		# Now embed the constructed flux profile at frequency f in the image.
		time_pos = int(xdrift[index])
		x0 = time_pos
		x1 = np.clip(time_pos+100, 0, img_sz-1)
		xlen = x1-x0

		img_decayx0 = time_pos
		img_decayx1 = time_pos+xlen
		img_risex0 = time_pos-flux.argmax()
		img_risex1 = time_pos
		
		#print(' %s %s %s %s'  %(img_decayx0, img_decayx1, img_risex0, img_risex1)  )
		deltx0 = img_decayx1 - img_decayx0
		deltx1 = img_risex1 - img_risex0

		try:
			if deltx0>1 and deltx1>1:
				image[f, img_decayx0:img_decayx1] = image[f, img_decayx0:img_decayx1]+flux[flux.argmax():flux.argmax()+xlen]*intensity
				image[f, img_risex0:img_risex1] = image[f, img_risex0:img_risex1]+flux[0:flux.argmax()]*intensity
			#else:
				# Sometimes this if loop is untrue for the whole burst and no burst is embedded. 
				# This else statement is to prevent return of bboxes for no bursts
				#no_embed+=1	
		except ValueError:	
			pdb.set_trace()
			embed_fail=True	

	boxx0 = t0-10
	boxy0 = f1
	lenx = 40
	leny = f0-f1
	bounding_box = [boxx0, boxy0, lenx, leny]
	
	return image, bounding_box, embed_fail


def embed_rfi(image, frange=[0,415], itensity=25):
	frfi = random.randint(frange[0], frange[1])
	sampling = random.randint(1,100)
	image[frfi, np.arange(0, np.shape(image)[0]-1, sampling)] = itensity

	return image


def embed_rfi_block(image, itensity=25):
	t0 = random.randint(1, np.shape(image)[0]-1)
	f0 = random.randint(1, np.shape(image)[0]-1)
	blocktsz = random.randint(2,50)
	blockfsz = random.randint(1,5)
	image[f0:f0+blockfsz, t0:t0+blocktsz] = itensity

	return image


def burst_cluster(i, img_sz):

	# This function is for choosing/randomising burst cluster characteristics e.g.,
	# whether ot not it's a single burst, a cluster or randomly distributed.
	cluster_t0=random.randint(10, img_sz-160)
	switcher={
		# Compact cluster
	    0:{'nbursts':random.randint(2, 4), 
		   'trange':[cluster_t0, cluster_t0+150], 
		   'head_range':[0,50],
		   'tail_range':[img_sz-50,img_sz-10]},
		# Random	   
	    1:{'nbursts':random.randint(2, 3), 
		   'trange':[5, img_sz-20], 
		   'head_range':[0,200],
		   'tail_range':[img_sz-50,img_sz-10]},
	    # Single		   
	    2:{'nbursts':1, 
		   'trange':[5, img_sz-20], 
		   'head_range':[0,200],
		   'tail_range':[img_sz-280,img_sz-5]}
	     }
	return switcher.get(i, "Invalid burst cluster")


def backsub(data):
    # Devide each spectrum by the spectrum with the minimum standard deviation.
    #data = np.log10(data)
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


########################################
#
#			Main procedure
#
#
if __name__=="__main__":
	
	# Produce an image, add background Gaussian noise and antenna frequency response
	img_sz = 512
	orig_image = np.zeros([img_sz, img_sz])
	nsamples = 5000
	backg = 1.0 + np.sin(np.linspace(0, np.pi, img_sz))*4
	backg = [backg, backg]
	backg = np.transpose(np.repeat(backg, img_sz/2, axis=0))
	tophat_kernel = Tophat2DKernel(2)
	root = '/Users/eoincarley/python/machine_learning/radio_burst_classifiers/Darknet/data/typeIII_cluster/'
	orig_image[::] = 1.0
	orig_image = orig_image + backg


	for img_index in np.arange(0, nsamples):
		image = copy(orig_image)

		randrfi = random.randint(40, 100)
		for i in np.arange(0, randrfi): image=embed_rfi(image, itensity=10)
		for i in np.arange(0, randrfi): image=embed_rfi(image, frange=[0, 80], itensity=25)
		for i in np.arange(0, 5): image=embed_rfi(image, frange=[250, 270], itensity=25)
		for i in np.arange(0, 5): image=embed_rfi(image, frange=[img_sz-30, img_sz-5], itensity=25)	
		randrfi = random.randint(0, 5)
		for i in np.arange(0, randrfi): image=embed_rfi_block(image, itensity=10)

		'''---------------------------------
		Following characteristics produce a cluster 
		of type IIIs together, as often occurs.
		'''
		cluster_type = 0 #random.randint(2,3)
		nbursts = burst_cluster( cluster_type, img_sz )['nbursts']
		trange = burst_cluster( cluster_type, img_sz )['trange']
		head_range = burst_cluster( cluster_type, img_sz )['head_range']
		tail_range = burst_cluster( cluster_type, img_sz )['tail_range']
		
		for i in np.arange(0, nbursts): 
			new_image, bbox, embed_fail = embed_typeIII(image, 
				trange=trange, head_range=head_range, tail_range=tail_range, intensity=random.uniform(1, 10))

			if not embed_fail:
				if i==0:
					bboxes = [bbox]
				else:
					bboxes.append(bbox)	
		

		image = convolve(image, tophat_kernel)
		image = backsub(image)
		
		#bboxes = [ np.zeros(4) ]
		#----------------------------------------------#
		#
		#    Write png that will be ingested by CNN
		#    
		fig = plt.figure(1, frameon=False, figsize=(4,4))
		ax = fig.add_axes([0, 0, 1, 1])
		ax.axis('off')
		ax.imshow(image, cmap=plt.get_cmap('gray'), vmin=image.min()*random.uniform(1.0, 2.0), vmax=image.max()*random.uniform(0.4, 1))

		#---------------------------------------------------------#
		#
		#	Make the coords of feature for Darknet/YOLO training
		#
		png_file = root+'image_'+str(format(img_index, '05'))+'.png'
		txt_file = root+'image_'+str(format(img_index, '05'))+'.txt' # Coordinates file for YOLO.
		file = open(txt_file, "w")
		for bbox in bboxes:
			#rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,edgecolor='r',facecolor='none')	
			#ax.add_patch(rect)
			xcen = bbox[0]+bbox[2]/2.0
			ycen = bbox[1]+bbox[3]/2.0
			dnet_coords = np.array([xcen, ycen, bbox[2], bbox[3]])/img_sz
			txt_coords = "0 "+str(dnet_coords[0])+" "+str(dnet_coords[1])+" "+str(dnet_coords[2])+" "+str(dnet_coords[3])+"\n"
			#print(txt_coords)
			file.write(txt_coords)

		
		#plt.show()	
		#pdb.set_trace()
		file.close()
		print('Saving %s' %(png_file))
		fig.savefig(png_file, transparent = True, bbox_inches = 'tight', pad_inches = 0, format='png')
		os.system("mogrify -format jpg -trim -resize 512x512 "+png_file)
		os.system("rm "+png_file)
		plt.close(fig)


