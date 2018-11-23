import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage.filters as smooth
from astropy.convolution import convolve, Tophat2DKernel



def embed_typeIII(image, trange=[10,490], head_range=[0,200], tail_range=[300,499]):

	t0 = random.randint(trange[0], trange[1])  	# Position of the type III head (top) in time.
	f0 = random.randint(300,499)    # Position of the type III head (top) in frequency.
	f1 = random.randint(head_range[0], head_range[1])		# Position of the type III tail (bottom) in frequency.
	nfreqs = f0-f1

	times = np.linspace(0, 499, 1000)
	frange = np.arange(int(f1), int(f0))
	driftrate = random.randint(1, 20) 		# In pixel units.
	xdrift = np.linspace(t0, t0-driftrate, len(frange))

	headsize = random.randint(1, 10)       						# Controls how much the head of the type III decays away -> how fat the head is.
	tspread0 = np.linspace(0, headsize, 10)  					# The first ten pixels of the head get fatter
	tspread1 = np.linspace(headsize, 0.1, len(frange[10::]))   	# After first ten pixels it get smaller and decays to 0.1
	tspread = np.concatenate((tspread0, tspread1))

	randpeak = random.randint(3,7)
	peak_flux_at_f = np.repeat(3, len(frange))


	# Controls the intensity along type III body. The tail is randomised in intensity to make it look inhomogeneous
	randpeak = random.randint(3,10)
	tail_indices = np.arange( int(len(frange)*0.4), int(len(frange)-1))
	peak_flux_at_f[tail_indices] = random.randint(0, randpeak, len(tail_indices))
	peak_flux_at_f = smooth.gaussian_filter1d(peak_flux_at_f, 3)
	#peak_flux_at_f = random.randint(0, randpeak, len(frange))   # For adding a randomness to the type III intensity along it's body.

	rise_time = 1.0  # Travelling across time for a particular frequency, 
					 # this controls how fast intensity rises. Basically 1 pixel.

    # This loop controls the rise and decay profile at each frequency.
	for index, f in enumerate(frange):

		fluxrise = 1.0/np.exp(times/1.0) 
		tdecay = tspread[index]  # Decay in time decreases as we go higher in frequency (towards the tail)
		fluxdecay = 1.0/(5.0*np.exp(-times/tdecay))
		flux = 1.0/(fluxrise + fluxdecay)

		flux = np.clip(flux*peak_flux_at_f[index], 0, 20)
		
		# Now embed the constructed flux profile at frequency f in the image.
		time_pos = int(xdrift[index])
		x0 = time_pos
		x1 = np.clip(time_pos+30, 0, 499)
		xlen = x1-x0


		img_decayx0 = time_pos
		img_decayx1 = time_pos+xlen
		img_risex0 = time_pos-flux.argmax()
		img_risex1 = time_pos
		
		image[f, img_decayx0:img_decayx1] = image[f, img_decayx0:img_decayx1]+flux[flux.argmax():flux.argmax()+xlen]
		image[f, img_risex0:img_risex1] = image[f, img_risex0:img_risex1]+flux[0:flux.argmax()]

	return image	


def embed_rfi(image, itensity=25):
	frfi = random.randint(0,499)
	sampling = random.randint(1,100)
	image[frfi, np.arange(0,499, sampling)] = itensity

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

image = np.zeros([500,500])
image[::] = 1.0

'''---------------------------------
 Following characteristics produce a cluster 
 of type IIs together, as often occurs.
'''
cluster_type = random.randint(0,3)
nbursts = burst_cluster( cluster_type )['nbursts']
trange = burst_cluster( cluster_type )['trange']
head_range = burst_cluster( cluster_type )['head_range']
tail_range = burst_cluster( cluster_type )['tail_range']

for i in np.arange(0, nbursts): 
	image=embed_typeIII(image, 
		trange=trange, head_range=head_range, tail_range=tail_range)

nrfi=15
for i in np.arange(0, nrfi): image=embed_rfi(image)
tophat_kernel = Tophat2DKernel(3)
image = convolve(image, tophat_kernel)
for i in np.arange(0, nrfi): image=embed_rfi(image, itensity=3)


fig = plt.figure(0)
plt.imshow(image, cmap='gray')
plt.show()
