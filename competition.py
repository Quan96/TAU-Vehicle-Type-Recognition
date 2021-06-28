# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#Create an index of class names
class_names = sorted(os.listdir(r"C:/Users/Ossi/Documents/PR and ML/Competition_data_unzip/train/train"))

#Prepare a pretrained CNN for feature extraction
base_model = tf.keras.applications.mobilenet.MobileNet(
input_shape = (224,224,3),
include_top = False)

tf.keras.utils.plot_model

in_tensor = base_model.inputs[0] # Grab the input of base model
out_tensor = base_model.outputs[0] # Grab the output of base model
# Add an average pooling layer (averaging each of the 1024 channels):
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)
# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs = [in_tensor],
outputs = [out_tensor])
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = "categorical_crossentropy", optimizer = "sgd")
path_to_data = path+os.sep+'vehicle'+os.sep+'train'+os.sep+'train'

# Find all image files in the data directory.
X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
for root, dirs, files in os.walk(path_to_data):
    for name in files:
        if name.endswith(".jpg"):
            # Load the image:
            img_name = root + os.sep + name
            print(img_name)
            img = plt.imread(img_name)
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128
            
            # Push the data through the model:
            x = model.predict(img[np.newaxis, ...])[0]
            # And append the feature vector to our list.
            X.append(x)
            # Extract class name from the directory name:
            label = img_name.split(os.sep, -2)
            y.append(class_names.index(label[1]))
# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

# experiment with the following classifiers
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

names = ["lda", "svmlinear", "svmrbf", "logreg", "forest"]
classifiers = [LinearDiscriminantAnalysis(), LinearSVC(), 
               SVC(kernel="rbf"), LogisticRegression(), RandomForestClassifier()]

accuracies = []

for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    accuracies.append(accuracy)
             