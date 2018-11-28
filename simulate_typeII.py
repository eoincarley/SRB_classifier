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
	t0 = random.randint(20, img_sz-220)
	t1 = random.randint(t0+200., img_sz-10)
	f0 = random.randint(20, img_sz-20)
	times = np.linspace(t0, t1, 300, dtype=int)

	drif_parm0 = random.uniform(1.2, 1.8)
	drif_parm1 = random.uniform(1.0, 1.8)
	drift_range = drif_parm0/np.linspace(1, 2, len(times))**drif_parm1

	timeszero = times - times[0]
	freqs = np.array(f0-timeszero*drift_range, dtype=int)

	x = np.linspace(0, 2*np.pi, len(times))
	
	env_phase1 = random.randint(1,4)
	env_phase2 = random.randint(1,4)
	env_phase3 = random.randint(1,4)
	env_phase4 = random.randint(1,4)
	env1 = np.abs(np.sin(x*env_phase1))
	env2 = np.abs(np.sin(x*env_phase2))

	nenvs = random.randint(1,7)
	for i in np.arange(1,nenvs):
		env1 = env1 + (1/i)*np.sin(i*x*env_phase3) 
		env2 = env2 + (1/i)*abs(np.sin(x*i*env_phase4))

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
	
	width_left = random.uniform(0.8, 2.0)
	width_right = random.uniform(0.8, 1.0)
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

def embed_rfi_block(image, itensity=25):
	t0 = random.randint(1,490)
	f0 = random.randint(1,490)
	blocktsz = random.randint(2,50)
	blockfsz = random.randint(1,5)
	image[f0:f0+blockfsz, t0:t0+blocktsz] = itensity

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
if __name__=="__main__":
	# Product an image, add background Gaussian noise and antenna frequency response

	nsamples = 3000
	img_sz=600
	for img_index in np.arange(0, nsamples):
		image = np.zeros([img_sz, img_sz])
		image[::] = 1.0

		image = image + 10.0*np.random.normal(1.0, 0.5, image.shape)
		backg = 1.0 + np.sin(np.linspace(0, np.pi, img_sz))*4
		backg = [backg, backg]
		backg = np.transpose(np.repeat(backg, img_sz/2, axis=0))
		image =  image + backg

		image = embed_typeII(image)
		image[np.isnan(image)]=1
		typeII_flux = random.randint(3,18)
		image = typeII_flux*image/image.max()
		image = image[50:550, 50:550]

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

		#fig = plt.figure(0)
		#vmax = 15 for no backsub
		#plt.imshow(image, vmin=0, vmax=20, cmap=plt.get_cmap('Spectral_r'))
		#plt.show()
		#-------------------------------------------------------------------#
		#
		#    Write png that will be ingested by Tensorflow trained model
		#    
		png_file = '/Users/eoincarley/python/machine_learning/radio_burst_classifiers/simulations/typeII/image_'+str(format(img_index, '04'))+'.png'
		print('Saving %s' %(png_file))
		fig = plt.figure(1, frameon=False, figsize=(4,4))
		ax = fig.add_axes([0, 0, 1, 1])
		ax.axis('off')
		ax.imshow(image, cmap=plt.get_cmap('gray'), vmin=1.0, vmax=15.0)
		fig.savefig(png_file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
		plt.close(fig)
