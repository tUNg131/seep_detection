# CGG Oil Seep Detection Exercise

## Objective
The objective is to produce a deep convolutional neural network (DCNN) model and a evaluation metric for image segmentation. 

## Dataset 
The given dataset contains 760 synthetic aperture radar images. To train the network, the dataset is randomly split 70:20:10 between training, validation and evaluation, respectively. 

## Run
To run the code, run the file ```./train.py``` with the datasets in the folder described above. The predictions are saved as ```.tif``` files in the ```output/``` folder. The best model is saved in the ```model/``` directory.

## Network Architecture
The architecture of choice for this task is the U-Net, which is a fully convolutional DCNN that have been used in biomedical imaging segmentation. 

## Loss Function
The loss function chosen for this task is the cross entropy loss. The reason this loss function is chosen is because the minimisation of the cross entropy corresponds to the maximum likelihood estimation of the network parameters given the dataset. This is assuming the datapoints are independent.

## Training Details
Training was done on batch size 32, using the Adam optimiser with learning rate 0.001. The maximum epoch was set at 100.

