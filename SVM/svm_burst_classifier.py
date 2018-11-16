#!/usr/bin/env python3

"""
 File:
    svm_burst_classifier.py

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
 	python3 svm_burst_classifier.py


 Version hitsory:

	Created 2018-May-05

"""


import pdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from PIL import Image


def svm_baseline(training_data, test_data):

	clf = svm.SVC(C=1, gamma=1, kernel='poly', cache_size=1000.0, class_weight='balanced')
	clf.fit(training_data[0], training_data[1])

	predictions = [int(a) for a in clf.predict(test_data[0])]
	num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
	print("%s of %s values correct." % (num_correct, len(test_data[1])))
	print("%s test set accuracy." % (num_correct/len(test_data[1])*100.))

	'''
	for a, y in zip(predictions, test_data[1]):
		#if a==y:
		print('Predicted burst type: %s' %(a))
		img = Image.open(test_files[i])
		data = np.asarray(img)
		fig = plt.figure()
		plt.imshow(data)
		plt.title('Predicted burst type: %s' %(a))
		plt.show()
		plt.pause(1)
		plt.close(fig)
		i=i+1
	'''
if __name__=="__main__":

	# The training and test data is constructed using build_train_data.py
	data=np.load('../train_test_data.npy')
	training_data = data[0]   
	test_data = data[1]    
	svm_baseline(training_data, test_data)