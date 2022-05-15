## 1. Assignment 3 - Transfer learning + CNN classification
In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction. 

Your ```.py``` script should minimally do the following:

- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report

## 2. Methods
This repository contains one script that performs image classification using the vgg16 model from tensorflow on the cifar10 dataset. In assignment 2, we saw that simpler models struggled to correctly classify the images in this dataset. vgg16 uses feature extraction and can take multiple array images with three color channels. We do therefore not need to reshape the data substantially, and therefore the model hopefully performs better.

## 3 Usage ```cnn_classification.py``` 
To run the code:
- Pull this repository with this file structure
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/cnn_classification.py -lr "learning rate of the model" -ds "decay steps of the model -dr "decay rate of the model -bs "batch size of the model" -e "epochs of the model" -vn "name of visualization to save" -rp "name of the report to save"```
  - The outputs in ```out``` were created with the following: ```python src/cnn_classification.py -lr 0.01 -ds 10000 -dr 0.9 -bs 128 -e 20 -vn "cnn_classification.jpg" -rp "cnn_report.txt"```

## 4. Discussion of Results
I played around with the different parameters can I found the code provided under ```Usage``` to be perform the best. The accuracy does not increase substantially beyond 20 epochs and the training and validation data start to diverge slightly.

The loss and accuracy of the training and validation data follow each other quite closely, so the model is not overfitting on the training data. However, the model still only achieves 52 % accuracy on the validation data with the parameters set to create the outputs in ```out```. A slower learning rate and more epochs might improve performance, since it seems from how the validation data and the training data follow each other that the model is learning something, but could perhaps use more time. 

There are also big differences on how the model performs on the individual labels. Compare cats that has a 39% accuracy and ships that has 63. Therefore, eventhough the model performs feature extract and we therefore do not need to reduce the images to 1-d arrays and convert to grayscale like in assignment 2, it is still difficult to correctly classify these images.
