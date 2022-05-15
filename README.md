## 1. Assignment 3 - Transfer learning + CNN classification

## 2. Methods
In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction. 

Your ```.py``` script should minimally do the following:

- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report

## 3 Usage ```cnn_classification.py``` 
To run the code:
- Pull this repository with this file structure
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/cnn_classification.py -lr "learning rate of the model" -ds "decay steps of the model -dr "decay rate of the model -bs "batch size of the model" -e "epochs of the model" -vn "name of visualization to save" -rp "name of the report to save"```
  - The outputs in ```out``` were created with the following: ```python src/CNN_classification.py -lr 0.01 -ds 10000 -dr 0.9 -bs 128 -e 10 -vn "cnn_classification.jpg" -rp "cnn_report.txt"```

## 4. Discussion of Results
