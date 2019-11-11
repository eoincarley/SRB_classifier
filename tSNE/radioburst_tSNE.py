#!/usr/bin/env python3

"""
 File:
    radioburst_tSNE.py

 Description:
    Perform t-distributed stochastic neighbour embedding on 
    RSTN radio bursts to see if they are separable in NxX vector space

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
 	python3 rradioburst_tSNE.py

 Version hitsory:

	Created 2018-May-05

"""

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":

	# The training and test data is constructed using build_train_data.py
	sns.set()
	data=np.load('../train_test_data_trial6.npy')
	training_data = data[0]   
	training_imgs = training_data[0]
	targets = training_data[1]


	sc = StandardScaler()  
	training_imgs = sc.fit_transform(training_imgs)  
	#-------------------------------------------------------;
	#
	#	
	projected=TSNE(n_components=2).fit_transform(training_imgs) # Project onto two dimensional space.
	ind0=np.where(targets==0)  # For separate plot commands to assign different color and label.
	ind2=np.where(targets==2)
	ind3=np.where(targets==3)

	plt.scatter(projected[ind0, 0], projected[ind0, 1],
            c='grey', edgecolor='none', alpha=0.5, label='No burst')
	plt.scatter(projected[ind3, 0], projected[ind3, 1],
            c='c', edgecolor='none', alpha=0.5, label='Type III')
	plt.scatter(projected[ind2, 0], projected[ind2, 1],
            c='darkmagenta', edgecolor='none', alpha=0.5, label='Type II')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.legend()
	plt.show()