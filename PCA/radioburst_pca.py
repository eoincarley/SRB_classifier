
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



if __name__=="__main__":

	# The training and test data is constructed using build_train_data.py
	sns.set()
	data=np.load('../train_test_data.npy')
	training_data = data[0]   
	training_imgs = training_data[0]
	targets = training_data[1]


	#-------------------------------------------------------;
	#
	#	PCA can separate somewhat type IIs and type IIs along
	#   two principle components, but the type IIs and 'no bursts' overlap
	#
	pca = PCA(2) # Find 2 PCA axes.
	projected = pca.fit_transform(training_imgs)  # Project data onto two PCA axes

	plt.figure(0)

	ind0=np.where(targets==0)  # For separate plot commands to assign different color and label.
	ind2=np.where(targets==2)
	ind3=np.where(targets==3)

	cols = ['grey']*len(ind0[0]) +['darkmagenta']*len(ind2[0]) +['c']*len(ind3[0]) 
	types = ['No burst']*len(ind0[0]) +['typeII']*len(ind2[0]) +['type III']*len(ind3[0]) 

	plt.scatter(projected[ind0, 0], projected[ind0, 1],
            c='grey', edgecolor='none', alpha=0.5, label='No burst')
	plt.scatter(projected[ind2, 0], projected[ind2, 1],
            c='darkmagenta', edgecolor='none', alpha=0.5, label='Type II')
	plt.scatter(projected[ind3, 0], projected[ind3, 1],
            c='c', edgecolor='none', alpha=0.5, label='Type III')

	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	plt.legend()
	plt.show()

	#----------------------------------------------------------------;
	#
	#	  Trying 3-dimensions. Variance only in large type III scatter
	#
	pca = PCA(3) 
	projected = pca.fit_transform(training_imgs)

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	
	x,y,z = 0,1,2   # Set PCA(3) to larger value. 
					# Looking at the higher dimensional components, 
					# it seems the classes are inseparable.
	ax.scatter(projected[ind0, x], projected[ind0, y], projected[ind0, z],
            c='grey', edgecolor='none', alpha=0.5, label='No burst')
	ax.scatter(projected[ind2, x], projected[ind2, y], projected[ind2, z],
            c='darkmagenta', edgecolor='none', alpha=0.5, label='Type II')
	ax.scatter(projected[ind3, x], projected[ind3, y], projected[ind3, z],
            c='c', edgecolor='none', alpha=0.5, label='Type III')
	
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('PCA 3')
	ax.legend()
	plt.show()
	

	#----------------------------------------------------------------;
	#
	#	Plot the explained variance with number of PCA components
	#   Looks like 90% of variance is explained within 500 PCA components.
	#   This means data is scattered to a large degree along many dimensions
	#   of its space. 
	pca = PCA().fit(training_imgs)
	plt.figure(2)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('Number of PCA components')
	plt.ylabel('Cumulative explained variance');
	plt.show()