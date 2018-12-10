#!/usr/bin/env python3

"""
 File:
    randforest_srb_classify.py

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
 	python3 randforest_srb_classify.py


 Version hitsory:

	Created 2018-May-05

"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def rf_baseline(training_data, test_data):

	x_train = training_data[0]
	y_train = training_data[1]
	x_test = test_data[0]
	y_test = test_data[1]

	sc = StandardScaler()  
	x_train = sc.fit_transform(x_train)  
	x_test = sc.transform(x_test)  

	classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
	classifier.fit(x_train, y_train) 

	y_pred = classifier.predict(x_test)  

	print(confusion_matrix(y_test,y_pred))  
	print(classification_report(y_test,y_pred))  
	print(accuracy_score(y_test, y_pred))  



if __name__=="__main__":

	# The training and test data is constructed using build_train_data.py
	data=np.load('../train_test_data.npy')
	training_data = data[0]   
	test_data = data[1]    
	rf_baseline(training_data, test_data)