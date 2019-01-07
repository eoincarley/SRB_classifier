# SRB_classifier
Training and development of machine learning algorithms to automatically classify solar radio burst images. 
The radio burst images are dynamic spectra of type II and III radio bursts from RSTN and simulated radio bursts.
This set of codes tests various machine learning algorithms' capability to be trained on such data and automatically
classify type II and III radio bursts. The goal is to train the algorithms on RSTN and simulated data and perform the
classification on I-LOFAR (IE613) data.

### Training and test data

The training and test data is a combination of radio burst observations from observed by the Radio Solar Telescope Network (RSTN) from 1996 to present and simulated dynamic spectra using simulate_typeIII.py and simulate_typeII.py. The simulation
codes were to make up for the lack of RSTN observations, which only amount to approximately 1000 of each class. An equal
number of 'no radio burst' images were made to cover true negative detections.

If you assemble your own data, only plot the dynamic spectrum itself. There should be no borders or axes etc.

### For SVM, Random Forest and PCA

For these algorithms the training and test data are assembled into the appropriate format (2D image -> 1D vector) using build_training_data.py. This searches for radio burst images in a local folder called 'radio_bursts'. Create subdirectories in here to the typeII, typeIII and empty dynamic spectra (type0). 

Once data is read into build_trainin_data it assembles all training and test data into a numpy array and saves it to train_test_data.npy. This .npy file is used by the SVM, RF, PCA and Keras scripts.

SVM and Random Forst show reasonable classifcation accuracy (74%), while a 1-hidden layer neural network built using Keras shows similar results. The PCA analysis that some type0, typeII and typeIII radio burst images are indistinguishable in an NxN vector space (where NxN is image dimension). Hence, the classical machine learning classification algorithms, as well as basic neural networks, may never be able to achieve high accuracy i.e., the radio bursts are not completely separable in the NxN vector space. t-Distributed stochastic neighbout embedding would also be a nice way to show this. It's likely that only a convolutional neural network would be capable of accurate classification. 

Similar classification accuracy was achieved with a 1-hidden layer neural network built using Keras. Generally, this
type of NN suffers from the same problem as the classical algorithms i.e., the radiobursts are not entirely separable in
the NxN vector space of pixel values. More sophisticated representations of the images are needed e.g, using convolutional
neural networks.

### For InceptionV3 and Darknet-YOLOv3

The input for the training of ImagenetV3 and Darkent are the images (dynamic soectra themselves) themselves. The way I've written the scripts which call these neuural nets, the images should be in a local file called radio/bursts with type0, typeII and typeIII subdirectories.

InceptionV3 can be called using Python-TensorFlow and, once trained on RSTN and the simulations, is adapetd into scripts to classify data from ILOFAR. Due to the lack of training data, transfer learning had to be used here. These means only the ~6000 param fully connected final layer of Imagenet is trained. Good results of 95% on the validation set using this algorithm. the transfer learning can be done on a standard desktop.

Darknet-YOLO is an image segmentation algorithm that I'm currently attempting to use to recognise solar radio bursts. If trained well, it should be able to output bounding boxes around the radio bursts. It could potentially be a powerfull and fast radio burst identifier in dynamic spectra. The full yolov3 network is large, with ~110 layers. It can be trained on custom objects using pretrained weights. It requires fairly hefty computational resources i.e. you will need access to a nice GPU of ~4 Gb RAM. In my case I'm training on a Google Cloud VM Instance on which I have access to an NVIDIA Tesla K80 GPU. Using a batch size of 64 images, it'll get though about 10,000 images in ~30 minutes. That should be enough to achieve some accurate results.

### Type III and type II simulator

The radio burst simulators are paramateric and not physics-based. They randomize as much as possible the number of radio bursts, the clustering, the drift rate, the intensity/inhomogineity etc. 
