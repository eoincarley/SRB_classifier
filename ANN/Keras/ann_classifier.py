import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

'''
----------------------------------------------------------------------
	Construct a simple neural network (1-hidden layer) using keras
	and train on RSTN data. Original images have been put into a 1e4 vector,
	so input dimension of same size naturally leads to large number of weights
	and biases to be learned. Due to the relatively small number of images here
	(~3000), undetermination and overfitting will be a problem. Hence the use of
	a decay param for the optimizer and use of Dropout. 

	Accurcay on the RSTN data set is ~75%, and generally quite similar to SGD and 
	Random Forest. No surprise, as the radio bursts can occur anywhere
	within the image, meaning the different types of bursts are not completely 
	separable in the 1e4 dimensional space (this can also be shown using the PCA
	analysis performed along with this package of experiments).

	A note on Nesterov accelerated gradient descent: The momentum parameter updates the weights based on 
	the accumulated gradient of previous steps (instead of just the local gradient, as in the normal SGD).
	The Nesterov approach updates the weights using the momentum of accumulated weights, but also taking into
	account the local gradient. The approach is; update weight using accumulated gradient -> correct for local
	gradient. This is a good way of preventing 'over-shooting' a minimum due to too much momentum in the approach to that
	minimum.
'''

model = Sequential()
sgd = SGD(lr=0.01, decay=1e-1, momentum=0.9, nesterov=True) 

model.add(Dense(100, activation='relu', input_dim=10000))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data=np.load('../../train_test_data.npy')
training_data = data[0]   
training_imgs = training_data[0]  # 1D vector representation of original RSTN dynamic spectra (images).
labels = training_data[1] 		  # Labels of 0, 2 and 3 depending on the radio burst type.
one_hot_labels = keras.utils.to_categorical(labels, num_classes=4) # One-hot ecoding for network training. Note there are no '1' labels. 
																   # Could optimize this to remove redundant output neuron.

model.fit(training_imgs, one_hot_labels, epochs=100, batch_size=32)

test_data = data[1]    
test_imgs = test_data[0]	# 1D vector representation of original RSTN dynamic spectra (images).
test_labels = test_data[1]  # Labels of 0, 2 and 3 depending on the radio burst type.
one_hot_test_labels = keras.utils.to_categorical(test_labels, num_classes=4)  # One hot encoding, same as before.

loss, acc = model.evaluate(test_imgs, one_hot_test_labels, batch_size=300)
print('Test loss:', loss)
print('Test accuracy:', acc)
