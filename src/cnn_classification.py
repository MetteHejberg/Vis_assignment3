# operating system
import os 
import sys

# tf tools
import tensorflow as tf

# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense) 
                                    
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

# import argparse
import argparse


# define our model
def mdl(learn_rate, dec_steps, dec_rate):
    # initialising the model and set parameters
    model = VGG16(include_top = False,
                  pooling = "avg",
                  input_shape = (32, 32, 3))
    # desabling convolutional layers 
    for layer in model.layers:
        layer.trainable = False
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)
    
    #define the new model
    model = Model(inputs = model.inputs,
                  outputs = output)
    
    #compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = learn_rate,
        decay_steps = dec_steps,
        decay_rate = dec_rate)
    # define learning rate
    sgd = SGD(learning_rate = lr_schedule)
    # compile
    model.compile(optimizer = sgd,
                  loss = "categorical_crossentropy",
                  metrics = ["accuracy"])
    return model

# a function that loads and processes our data
def load_process_data():
    # load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # normalize
    X_train_norm = X_train/255
    X_test_norm = X_test/255
    
    # create one-hot encodings
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    # initialize label names for CIFAR-10 dataset
    label_names = ["airplane", 
                   "automobile", 
                   "bird", 
                   "cat", 
                   "deer", 
                   "dog", 
                   "frog", 
                   "horse", 
                   "ship", 
                   "truck"]
    return X_train_norm, X_test_norm, y_train, y_test, label_names

#train the model and return H
def train_model(model, X_train_norm, y_train, X_test_norm, y_test, b_size, eps):
    # fit the data to the model
    H = model.fit(X_train_norm, y_train,
                  validation_data = (X_test_norm, y_test),
                  batch_size = b_size,
                  epochs = eps,
                  verbose = 1)
    return H

# define plotting function
def plot_history(H, eps, vis_name): 
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, eps), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, eps), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, eps), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, eps), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    # save figure
    outpath_vis = os.path.join("out", vis_name)
    fig = plt.savefig(outpath_vis, dpi = 300, bbox_inches = "tight")
    

# a function that gets and saves the classification report
def classification_rep(model, X_test_norm, y_test, rep_name, label_names):
    # get predictions
    predictions = model.predict(X_test_norm, 128) 
    # make the classification report
    report = classification_report(y_test.argmax(axis = 1), predictions.argmax(axis = 1), target_names = label_names)
    # print report
    print(report)
    # create outpath
    p = os.path.join("out", rep_name)
    # save report as txt file
    sys.stdout = open(p, "w")
    text_file = print(report)

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-e", "--eps", type=int, required=True, help="the number of epochs")
    ap.add_argument("-vn", "--vis_name", required=True, help="the name of the figure")
    ap.add_argument("-lr", "--learn_rate", type=float, required=True, help="the learning rate of the model")
    ap.add_argument("-ds", "--dec_steps", type=int, required=True, help="the decay steps of the model")
    ap.add_argument("-dr", "--dec_rate", type=float, required=True, help="the decay rate of the model")
    ap.add_argument("-bs", "--b_size", type=int, required=True, help="the batch size of the model")
    ap.add_argument("-rp", "--rep_name", required=True, help="the name of the classification report")
    args = vars(ap.parse_args())
    return args

# let's get the code to run!
def main():
    args = parse_args()
    model = mdl(args["learn_rate"], args["dec_steps"], args["dec_rate"])
    X_train_norm, X_test_norm, y_train, y_test, label_names = load_process_data()
    H = train_model(model, X_train_norm, y_train, X_test_norm, y_test, args["b_size"], args["eps"])
    fig = plot_history(H, args["eps"], args["vis_name"])
    report_df = classification_rep(model, X_test_norm, y_test, args["rep_name"], label_names)
    
if __name__ == "__main__":
    main()
    

