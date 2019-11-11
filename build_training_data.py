#!/usr/bin/env python3

"""
 File:
    build_train_data.py

 Description:
    Read in images of solar radio bursts. Smooth, intensity scale, rebin then
    reshape into a 1-dimensional vector. 

    The test data is necessarily given the same format.

    Results in training inputs and expected outpurs for use in machine learning 
    algorithms. Used for input in SVM, Random Forest, FFNN.

    Data may have to be in slightly different formats depend on ML Python
    package e.g., Scikit, PyTorch or Keras. Format produced by this code is
    nice for SVM from Scikit learn, but a bit of unpacking is required
    for Keras input.

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
 	python build_training_data.py --ntest 200


 Version hitsory:

	Created 2018-May-05


"""
import glob
import numpy as np
import pdb
from PIL import Image
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

#-------------------#
#
#	 Functions
# 
def build_1Dimg(files, burst_type):
	size = 50 #100
	# Go through all images, resize, flaten to 1D vector, normalise, concatenate
	for index, file in enumerate(files):
		img = Image.open(file)
		#img.thumbnail(size)
		#pdb.set_trace()
		data = np.asarray(img)
		data = data[0:498,1:499,0] # IDL saves a 1 pxel white border. Get rid.
		data= resize(data, (size, size))
		data = data.flatten()
		#data = data/np.max(data)
		# Rescale data values to be zero mean and standard dev 1. Good for ML algos.
		data = StandardScaler().fit_transform(data.reshape(-1,1)).flatten()
		data[np.isnan(data)]=0
		if index==0:
			imgs1D=[data]
		else:	
			imgs1D=np.concatenate((imgs1D, [data]))
		print('Reformatted %s' %(file))
	types = np.repeat(burst_type, imgs1D.shape[0]) # Labels for the expect output given an image
	return imgs1D, types		


def assemble_train_data(files0, files2, files3):
	# Assemble training (and test) data into 1D images and their corresponding burst types
	type0_1D, type0 = build_1Dimg(files0, 0)
	typeII_1D, typeII = build_1Dimg(files2, 2)
	typeIII_1D, typeIII = build_1Dimg(files3, 3)
	types_1Dimgs = np.concatenate((type0_1D, typeII_1D, typeIII_1D))
	types = np.concatenate((type0, typeII, typeIII))
	assembled_data = (types_1Dimgs, types)
	return assembled_data


if __name__=="__main__":

	from optparse import OptionParser
	o=OptionParser()
	o.add_option('-T', '--ntest', dest='ntest', default=100, type=int)
	o.add_option('-s', '--save',  action="store_true", default=False, dest='save', 
		help = 'Need to give True inorder to specifcally save/overwrite the training-test file.')
	opts, args = o.parse_args()
	ntest = opts.ntest
	save = opts.save
	# ntest is number of test images for each category. So with the three categories here
	# there are 300 test images by default. 

	# Following images are cosntructed from dynamic spectra downloaded from RSTN archives.
	# Burst times and types were from the SWPC event lists. List not always accurate, so watch
	# out for mislabels.

	# root is normally radio_bursts, which is linked to one of the trials in ~/Data/RSTN_archive.
	root = 'radio_bursts'

	type0files = np.sort(glob.glob(root+'/type0/*.*'))
	type0files_train = type0files[0:-ntest]
	type0files_test = type0files[-ntest::]

	typeIIfiles = np.sort(glob.glob(root+'/typeII/*.*'))
	typeIIfiles_train = typeIIfiles[0:-ntest]
	typeIIfiles_test = typeIIfiles[-ntest::]

	typeIIIfiles = np.sort(glob.glob(root+'/typeIII/*.*'))
	typeIIIfiles_train = typeIIIfiles[0:-ntest]
	typeIIIfiles_test = typeIIIfiles[-ntest::]

	training_data = assemble_train_data(type0files_train, typeIIfiles_train, typeIIIfiles_train)
	test_data = assemble_train_data(type0files_test, typeIIfiles_test, typeIIIfiles_test)
	
	data = np.array([training_data, test_data])
	if save==True: np.save('train_test_data_trial6.npy', data)

	print('Training and test data assembly complete.')
	ntraining_imgs = np.shape(data[0][1])[0]
	ntest_imgs = np.shape(data[1][1])[0]
	print('- Assembled %s training images' %(ntraining_imgs))
	print('- Assembled %s test images' %(ntest_imgs))

	print('Training data breakdown:')
	ntype0 = len(np.where(data[0][1]==0)[0])
	ntype2 = len(np.where(data[0][1]==2)[0])
	ntype3 = len(np.where(data[0][1]==3)[0])
	print('- No burst: %s' %(ntype0))
	print('- Type II burst: %s' %(ntype2))
	print('- Type III burst: %s' %(ntype3))

