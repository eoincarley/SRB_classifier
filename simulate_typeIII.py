import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pdb



def embed_typeIII(image):

	t0 = random.randint(0,499)  # Position of the type III head (top).
	f0 = random.randint(300,499)
	f1 = random.randint(0,200)
	nvals = f0-f1
	fluxpix = np.linspace(1, 4, nvals)
	#image[f1:f0, t0] = fluxpix[::-1]

	times = np.linspace(0,499,1000)
	frange = np.arange(int(f1), int(f0))
	driftrate = random.randint(1,5)
	xdrift = np.linspace(t0, t0-driftrate, len(frange))

	headdecay = random.randint(1,10)
	tspread0 = np.linspace(0, headdecay, 10)
	tspread1 = np.linspace(headdecay, 0.1, len(frange[10::]))
	tspread = np.concatenate((tspread0, tspread1))

	auxpeak = random.randint(2,5)
	peak_flux_at_f = random.randint(1, auxpeak, len(frange))

	for index, f in enumerate(frange):

		fluxrise = 1.0/(1.0*np.exp(times/1.0))
		tdecay = tspread[index]
		fluxdecay = 1.0/(5.0*np.exp(-times/tdecay))
		flux = 1.0/(fluxrise + fluxdecay)

		flux = np.clip(flux*peak_flux_at_f[index], 1, 20)

		time_pos = int(xdrift[index])
		x0 = time_pos
		x1 = np.clip(time_pos+30, 0, 499)
		xlen = x1-x0
		image[f, time_pos:time_pos+xlen] = flux[flux.argmax():flux.argmax()+xlen]
		image[f, time_pos-flux.argmax():time_pos] = flux[0:flux.argmax()]

	return image	


image = np.zeros([500,500])
image[::] = 1.0

image=embed_typeIII(image)
image=embed_typeIII(image)
image=embed_typeIII(image)
image=embed_typeIII(image)
image=embed_typeIII(image)


fig = plt.figure(0)
plt.imshow(image)
plt.show()
