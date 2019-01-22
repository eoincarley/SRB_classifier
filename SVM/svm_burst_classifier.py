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
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def svm_train_eval(training_data, test_data):

	x_train = training_data[0]
	y_train = training_data[1]
	x_test = test_data[0]
	y_test = test_data[1]

	sc = StandardScaler()  
	x_train = sc.fit_transform(x_train)  
	x_test = sc.transform(x_test)  

	clf = svm.SVC(C=1, gamma=1, kernel='poly', cache_size=1000.0, class_weight='balanced')
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)  

	#---------------------------------------------------#
	#		Compare predicted to test y values 
	#		to evaluate various performance metrics.
	test_acc = accuracy_score(y_test, y_pred)
	print('Test set accurcay: %s' %(test_acc))  
	print(classification_report(y_test, y_pred))  
	cmat = confusion_matrix(y_test, y_pred)  

	fig = plt.figure()
	sns.heatmap(cmat.T, square=True, annot=True, fmt='d', cbar=False)
	plt.xlabel('True')
	plt.ylabel('Predicted')
	plt.xticks(np.arange(3)+0.5, ('No burst', 'Type II', 'Type III'))
	plt.yticks(np.arange(3)+0.5, ('No burst', 'Type II', 'Type III'))
	plt.tick_params(labelsize=8)
	plt.show()


if __name__=="__main__":

	# The training and test data is constructed using build_train_data.py
	data=np.load('../train_test_data.npy')
	training_data = data[0]   
	test_data = data[1]    
	svm_train_eval(training_data, test_data)