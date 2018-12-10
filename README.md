# SRB_classifier
Training and development of machine learning algorithms to automatically classify solar radio bursts

### Training and test data

The training and test data is a combination of radio burst observations from observed by the Radio Solar Telescope Network (RSTN) and simulated dynamic spectra using simulate_typeIII.py and simulate_typeII.py. 

If you assemble your own data, only plot the dynamic spectrum itself. There should be no borders or axes etc.

### For SVM, Random Forest and PCA

For these algorithms the training and test data are assembled into the appropriate format (2D image -> 1D vector) using build_training_data.py. This searches for radio burst images in a local folder called 'radio_bursts'. Create subdirectories in here to
the typeII, typeIII and empty dynamic spectra (in my case call this type0). 

Once data is read into build_trainin_data it assembles all training and test data into a numpy array and saves it to train_test_data.npy. This .npy file is used by the SVM, RF and PCA scripts.

While SVM and Random Forst show reasonable classifcation accuracy (74%), the PCA analysis that type0, typeII and typeIII radio burst images are indistinguishable when used in an NxN vector space (where NxN is image dimension). The classical machine learning classification algorithms that attempt to separate these classes in the vector space may never be able to achieve high accuracy. It's likely that only a convolutional neural network would be capable of accurate classification. 

### For ImagenetV3 and Darknet

The input for the training of ImagenetV3 and Darkent are the images (dynamic soectra themselves) themselves. The way I've written the scripts which call these neuural nets, the images should be in a local file called radio/bursts with type0, typeII and typeIII subdirectories.

### Type III and type II simulator

The radio burst simulators are paramateric and not physics-based. They randomize as much as possible the number of radio bursts, the clustering, the drift rate, the intensity/inhomogineity etc. 
