import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage.filters as smooth
from scipy.stats import norm
from astropy.convolution import convolve, Tophat2DKernel
from simulate_typeIII import embed_typeIII, burst_cluster
poly=np.polynomial.polynomial.polyval


def embed_typeII(image):

	img_sz = np.shape(image)[0]
	t0 = random.randint(20, img_sz/2-10)
	t1 = random.randint(img_sz/2+10, img_sz-10)
	f0 = random.randint(20, img_sz-20)
	times = np.linspace(t0, t1, 300, dtype=int)

	drift_range = 1.2/np.linspace(1, 2, len(times))**1.0

	timeszero = times - times[0]
	freqs = np.array(f0-timeszero*drift_range, dtype=int)

	x = np.linspace(0, 2*np.pi, len(times))
	
	env1 = np.abs(np.sin(x*4))
	env2 = np.abs(np.sin(x*4))
	for i in np.arange(1,5):
		env1 = env1 + (1/i)*np.sin(i*x*3) 
		env2 = env2 + (1/i)*abs(np.sin(x*i*2))

	env1 = env1-env1.min()
	env1 = np.clip(env1/env1.max(), 0.01, 1)
	env2 = env2-env2.min()
	env2 = np.clip(env2/env2.max(), 0.01, 1)


	flux_env = np.zeros(len(times))
	flux_env[10:-10] = 2.0
	flux_env = smooth.gaussian_filter1d(flux_env, 50)
	flux_env1 = flux_env*env1
	flux_env2 = flux_env*env2
	

	bw_env = np.zeros(len(times))
	bw_env[5:-5] = 2.0
	bw_env = smooth.gaussian_filter1d(bw_env, 6)
	scales1 = bw_env*env1*5
	scales2 = bw_env*env2*5
	
	width_left = random.uniform(0.1, 2.5)
	width_right = random.uniform(0.1, 1.0)
	typeIII_scales = np.linspace(width_left, width_right, len(times))		
	
	hdispl = np.linspace(150, 100, len(times))	
	margin1 = img_sz-10
	margin0 = 10	
	
	for index in np.arange(0, len(times), 1):
		t = times[index]
		f = freqs[index]
		h = f + hdispl[index]
		scl = typeIII_scales[index]
		scl0 = -80*scl
		scl1 = -30*scl
		scl2 = 20*scl
		scl3 = 50*scl

		hhead0 = h+scl0
		hhead1 = h+scl1
		htail0 = h+scl2
		htail1 = h+scl3

		fhead0 = f+scl0
		fhead1 = f+scl1
		ftail0 = f+scl2
		ftail1 = f+scl3

		if ftail1<margin1 and fhead0>margin0 and t>margin0 and t<margin1:
			image=embed_typeIII(image, trange=[t,t+1], head_range=[fhead0, fhead1], tail_range=[ftail0, ftail1], intensity=flux_env1[index])
			
		if htail1<margin1 and hhead0>margin0 and t>margin0 and t<margin1:
			image=embed_typeIII(image, trange=[t,t+1], head_range=[hhead0, hhead1], tail_range=[htail0, htail1], intensity=flux_env2[index])

	tophat_kernel = Tophat2DKernel(width_left)
	image = convolve(image, tophat_kernel)
	return image	


def embed_rfi(image, frange=[0, 499], itensity=25):
	img_sz = np.shape(image)[0]
	frfi = random.randint(frange[0], frange[1])
	sampling = random.randint(1,100)
	image[frfi, np.arange(0, img_sz-1, sampling)] = itensity

	return image




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
# Product an image, add background Gaussian noise and antenna frequency response
img_sz=600
image = np.zeros([img_sz, img_sz])
image[::] = 1.0

image = image + 10.0*np.random.normal(1.0, 0.5, image.shape)
backg = 1.0 + np.sin(np.linspace(0, np.pi, img_sz))*4
backg = [backg, backg]
backg = np.transpose(np.repeat(backg, img_sz/2, axis=0))
image =  image + backg


image = embed_typeII(image)
image[np.isnan(image)]=1
typeII_flux = random.randint(0,20)
image = typeII_flux*image/image.max()


'''---------------------------------
 Following characteristics produce a cluster 
 of type IIIs together, as often occurs.

cluster_type = random.randint(0,3)
nbursts = burst_cluster( cluster_type )['nbursts']
trange = burst_cluster( cluster_type )['trange']
head_range = burst_cluster( cluster_type )['head_range']
tail_range = burst_cluster( cluster_type )['tail_range']


for i in np.arange(0, nbursts): 
	image=embed_typeIII(image, 
		trange=trange, head_range=head_range, tail_range=tail_range)
'''

image = image[50:550, 50:550]
for i in np.arange(0, 60): image=embed_rfi(image, frange=[0, 80], itensity=20)
for i in np.arange(0, 5): image=embed_rfi(image, frange=[250, 270], itensity=15)
for i in np.arange(0, 10): image=embed_rfi(image, frange=[400, 450], itensity=10)
tophat_kernel = Tophat2DKernel(3)
image = convolve(image, tophat_kernel)
for i in np.arange(0, 100): image=embed_rfi(image, itensity=10)

image = backsub(image)

fig = plt.figure(0)
#vmax = 15 for no backsub
plt.imshow(image, vmin=0, vmax=20, cmap=plt.get_cmap('Spectral_r'))
plt.show()
