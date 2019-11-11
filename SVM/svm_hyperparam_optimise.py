"""
 File:
    svm_burst_hperparam_optimise.py

 Description:
    Run the SVM classification for a range C and gamma paramaters.
    See description of parameters below.

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
	C is a regularising hyperparameter. The larger C, the greater the effect of the
	hinge-loss function. This means that the SVM will not allow points to fall into the
	margin, resulting in an irregular hypersurface that 'avoids' the points i.e., overfitting.
	The smaller C, the less effective the hinge-loss function. Points are allowed to fall into
	the margin and the hypersurface can be smoother i.e., overfitting is prevented. 

	Gamma is a hyperparamater that determines whether or not the hyperplane definition
	is defined locally. Low gamma means hyperplane is defined taking into account
	points that are far from the support vectors e.g., the hyperplane is globally defined. High 
	gamma means the hyperplane will be more locally defined (type of overfitting). 
	This means gamma is the inverse of the radius of influence of a support vector. 
	This comes from the fact that the quadratic minimisation problem (which results in 
	the magrin boundary definition) depends on the dot-product of pairs of points (dot-products 
	through the Kernel). If gamma is large, only local dot-products have influence, 
	wheras small gamma means distant dot-producrs have influence. In this way, small gamma
	can regularise the training and prevent overfitting.

	This code calculates how accuracy varies with C and gamma.

 Examples:
 	python3 svm_burst_classifier.py


 Version hitsory:

	Created 2018-May-05
	v0.1 - Eoin Carley 

"""

import matplotlib.pyplot as plt
import pdb
from sklearn import svm

#-------------------#
#
#	 Functions
# 
def svm_baseline(training_data, test_data, C, gamm):
	print('   ')
	print('Training SVM with hyperparameters C=%s and gamma=%s' % (C, gamm))
	clf = svm.SVC(C=C, gamma=gamm, kernel='poly')
	clf.fit(training_data[0], training_data[1])

	predictions = [int(a) for a in clf.predict(test_data[0])]
	num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
	print("%s of %s values correct." % (num_correct, len(test_data[1])))
	accuracy = (num_correct/len(test_data[1]))*100
	print("%s Accuracy." % (accuracy))
	print('   ')
	return accuracy


if __name__=="__main__":

	# --------------------------------#	
	# Load in training and test data
	#	
	data=np.load('../train_test_data.npy')
	training_data = data[0]   
	test_data = data[1]  

	# --------------------------------#	
	# Define large set of ranges for C and gamma
	#
	nrange=13	  
	crange = np.logspace(-2, 10, nrange)
	gamma_range = np.logspace(-9, 3, nrange)
	gammaC_array = np.ndarray(shape=(nrange, nrange), dtype='float')

	# -----------------------------------------------#	
	#   Evaluate all gamma and C and fill the array
	#
	#for i, C in enumerate(crange):
	#	for j, gamm in enumerate(gamma_range):
	#		gammaC_array[i,j]=svm_baseline(training_data, test_data, C, gamm)

	gammaC_array=np.load('hyperparam_optimise.npy')

	# --------------------------------#	
	#       Plot the data
	#	
	xticklabels = ["{0:0.1e}".format(label) for label in crange]
	fig = plt.figure()
	ax = fig.add_axes([0.1, 0.17, 0.8, 0.76])
	img=ax.imshow(gammaC_array, cmap='plasma')
	ax.set_xlabel('C')
	ax.set_ylabel('Gamma')
	cb=fig.colorbar(img)
	cb.set_label("Percent accuracy on test data")

	plt.title('SVM percent accuracy; quadratic kernel')
	plt.xticks(np.arange(len(crange)), xticklabels, rotation=45)
	plt.yticks(np.arange(len(gamma_range)), gamma_range)
	plt.show()

