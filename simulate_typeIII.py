import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb

import scipy.ndimage.filters as smooth
from astropy.convolution import convolve, Tophat2DKernel
from numpy import random


def embed_typeIII(image, trange=[10,490], head_range=[0,200], tail_range=[300,499], intensity=1):

	img_sz = np.shape(image)[0]
	t0 = random.randint(trange[0], trange[1])  	# Position of the type III head (top) in time.
	f0 = random.randint(tail_range[0], tail_range[1])    # Position of the type III tail (bottom) in frequency.
	f1 = random.randint(head_range[0], head_range[1])		# Position of the type III head (top) in frequency.
	nfreqs = f0-f1

	times = np.linspace(0, img_sz-1, 1000)
	frange = np.arange(int(f1), int(f0))
	drift_range = np.linspace(1, 5, len(frange))
	base_driftrate = random.randint(10,30)
	xdrift = t0 + base_driftrate/drift_range 		# In pixel units. Responsible for how 'curved' the head is.
	#xdrift = np.linspace(t0, t0-driftrate, len(frange))

	headsize = random.randint(1, 10) 
	tailsize = random.uniform(0.1, 3)       						# Controls how much the head of the type III decays away -> how fat the head is.
	tspread0 = np.linspace(0, headsize, 20)  					# The first ten pixels of the head get fatter
	tspread1 = np.linspace(headsize, tailsize, len(frange[20::]))   	# After first ten pixels it get smaller and decays to 0.1
	tspread = np.concatenate((tspread0, tspread1))


	# Controls the intensity along type III body. The tail is randomised in intensity to make it look inhomogeneous
	randpeak = random.randint(3,7)
	peak_flux_at_f = np.repeat(5, len(frange))
	randpeak = random.randint(3,10)
	tail_indices = np.arange( int(len(frange)*0.4), int(len(frange)-1))
	peak_flux_at_f[tail_indices] = random.randint(0, randpeak, len(tail_indices))
	peak_flux_at_f = smooth.gaussian_filter1d(peak_flux_at_f, 6)

	envelope=np.linspace(random.uniform(0.3, 1), 1, len(peak_flux_at_f)) # Sometimes the type III will fade as it increases in frequency.
	peak_flux_at_f = envelope*peak_flux_at_f
	#peak_flux_at_f = random.randint(0, randpeak, len(frange))   # For adding a randomness to the type III intensity along it's body.

	rise_time = 1.0  # Travelling across time for a particular frequency, 
					 # this controls how fast intensity rises. Basically 1 pixel.

	# Following loop controls the rise and decay profile at each frequency.
	fluxrise = 1.0/np.exp(times/1.0) 
	fluxdecay_range = 1/np.linspace(1, 5, len(frange))
	for index, f in enumerate(frange):

		tdecay = tspread[index]  # Decay in time decreases as we go higher in frequency (towards the tail)
		fluxdecay = 1.0/(4.0*np.exp(-times/tdecay))
		flux = 1.0/(fluxrise + fluxdecay)

		flux = np.clip(flux*peak_flux_at_f[index], 0, 20)
		
		# Now embed the constructed flux profile at frequency f in the image.
		time_pos = int(xdrift[index])
		x0 = time_pos
		x1 = np.clip(time_pos+30, 0, img_sz-1)
		xlen = x1-x0

		img_decayx0 = time_pos
		img_decayx1 = time_pos+xlen
		img_risex0 = time_pos-flux.argmax()
		img_risex1 = time_pos
		
		#print(' %s %s %s %s'  %(img_decayx0, img_decayx1, img_risex0, img_risex1)  )
		deltx0 = img_decayx1 - img_decayx0
		deltx1 = img_risex1 - img_risex0

		if deltx0>1 and deltx1>1:
			image[f, img_decayx0:img_decayx1] = image[f, img_decayx0:img_decayx1]+flux[flux.argmax():flux.argmax()+xlen]*intensity
			image[f, img_risex0:img_risex1] = image[f, img_risex0:img_risex1]+flux[0:flux.argmax()]*intensity


	boxx0 = t0-5
	boxy0 = f1
	lenx = 40
	leny = f0-f1
	bounding_box = [boxx0, boxy0, lenx, leny]
	
	return image, bounding_box 	


def embed_rfi(image, frange=[0,499], itensity=25):
	frfi = random.randint(frange[0], frange[1])
	sampling = random.randint(1,100)
	image[frfi, np.arange(0,499, sampling)] = itensity

	return image


def embed_rfi_block(image, itensity=25):
	t0 = random.randint(1,490)
	f0 = random.randint(1,490)
	blocktsz = random.randint(2,50)
	blockfsz = random.randint(1,5)
	image[f0:f0+blockfsz, t0:t0+blocktsz] = itensity

	return image


def burst_cluster(i):

	# This function is for choosing/randomising burst cluster characteristics e.g.,
	# whether ot not it's a single burst, a cluster or randomly distributed.
	cluster_t0=random.randint(15, 350)
	switcher={
		# Compact cluster
	    0:{'nbursts':random.randint(2, 7), 
	    		   'trange':[cluster_t0, cluster_t0+120], 
	    		   'head_range':[0,50],
	    		   'tail_range':[480,499]},
		# Random	   
	    1:{'nbursts':random.randint(2, 10), 
	    		   'trange':[15, 480], 
	    		   'head_range':[0,200],
	    		   'tail_range':[300,499]},
	    # Single		   
	    2:{'nbursts':1, 
	    		   'trange':[15, 480], 
	    		   'head_range':[0,200],
	    		   'tail_range':[300,499]}
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
	
	# Product an image, add background Gaussian noise and antenna frequency response
	img_sz = 500
	image = np.zeros([img_sz, img_sz])
	nsamples = 5000

	for img_index in np.arange(0, nsamples):
		image[::] = 1.0

		#image = image+np.random.normal(1, 0.5, image.shape)
		backg = 1.0 + np.sin(np.linspace(0, np.pi, img_sz))*4
		backg = [backg, backg]
		backg = np.transpose(np.repeat(backg, img_sz/2, axis=0))
		image =  image + backg

		'''---------------------------------
		 Following characteristics produce a cluster 
		 of type IIIs together, as often occurs.
		'''
		cluster_type = random.randint(1,3)
		nbursts = burst_cluster( cluster_type )['nbursts']
		trange = burst_cluster( cluster_type )['trange']
		head_range = burst_cluster( cluster_type )['head_range']
		tail_range = burst_cluster( cluster_type )['tail_range']


		for i in np.arange(0, nbursts): 
			image, bbox = embed_typeIII(image, 
				trange=trange, head_range=head_range, tail_range=tail_range)
			if i==0:
				bboxes = [bbox] 
			else:
				bboxes.append(bbox)	
			
		
		randrfi = random.randint(0, 10)
		for i in np.arange(0, randrfi): image=embed_rfi(image, itensity=10)
		for i in np.arange(0, randrfi): image=embed_rfi(image, frange=[0, 80], itensity=25)
		for i in np.arange(0, 5): image=embed_rfi(image, frange=[250, 270], itensity=25)
		for i in np.arange(0, 5): image=embed_rfi(image, frange=[400, 450], itensity=25)	
		randrfi = random.randint(0, 5)
		for i in np.arange(0, randrfi): image=embed_rfi_block(image, itensity=10)
		tophat_kernel = Tophat2DKernel(2)
		image = convolve(image, tophat_kernel)
		image = backsub(image)

		#vmax = 15 for no backsub
		#plt.imshow(image, vmin=1.0, vmax=4, cmap=plt.get_cmap('Spectral_r'))
		#plt.show()
		#-------------------------------------------------------------------#
		#
		#    Write png that will be ingested by Tensorflow trained model
		#    
		root = '/Users/eoincarley/python/machine_learning/radio_burst_classifiers/Darknet/darknet/data/typeIII/'
		png_file = root+'image_'+str(format(img_index, '04'))+'.png'
		txt_file = root+'image_'+str(format(img_index, '04'))+'.txt'
		print('Saving %s' %(png_file))
		fig = plt.figure(1, frameon=False, figsize=(4,4))
		ax = fig.add_axes([0, 0, 1, 1])
		ax.axis('off')
		ax.imshow(image, cmap=plt.get_cmap('gray'), vmin=1.0, vmax=4.0)

		#--------------------------------------------#
		#	Make the coord of feature for DarkNet
		#
		file = open(txt_file, "w")
		for bbox in bboxes:
			#rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,edgecolor='r',facecolor='none')	
			#ax.add_patch(rect)
			xcen = bbox[0]+bbox[2]/2.0
			ycen = bbox[1]+bbox[3]/2.0
			dnet_coords = np.array([xcen, ycen, bbox[2], bbox[3]])/img_sz
			txt_coords = "0 "+str(dnet_coords[0])+" "+str(dnet_coords[1])+" "+str(dnet_coords[2])+" "+str(dnet_coords[3])+"\n"
			print(txt_coords)
			file.write(txt_coords)
		
		file.close()

		fig.savefig(png_file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
		plt.close(fig)


