# -*- coding: utf-8 -*-


import os
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

path = os.getcwd()
path_to_train = path+os.sep+'vehicle'+os.sep+'train'+os.sep+'train'
path_to_test = path+os.sep+'vehicle'+os.sep+'test'+os.sep+'testset'

class_names = sorted(os.listdir(path_to_train))

X = list()
y = list()
for root, dirs, files in os.walk(path_to_train):
    for name in files:
        if name.endswith(".jpg"):
            img_name = root + os.sep + name
            #print(img_name)
            image = Image.open(img_name)
            image = image.resize((224,224))
            image_array = np.asarray(image)
            image_array = image_array.astype(np.float16)
            hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            hoz_flip_array = np.asarray(hoz_flip)
            hoz_flip_array = hoz_flip_array.astype(np.float16)
            X.append(image_array)
            X.append(hoz_flip_array)
            label = img_name.split(os.sep, -2)
            #label = "Ambulance"
            y.append(class_names.index(label[-2]))
            #redo for augmented ones
            y.append(class_names.index(label[-2]))

X = np.asarray(X)
y = np.asarray(y)
y = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)


Base_model = MobileNet(include_top = False, input_shape = (224,224,3), 
                       weights = 'imagenet')

w = Base_model.output

w = Flatten()(w)
w = Dense(100, activation = 'relu')(w)
w = Dense(17, activation = 'softmax')(w)

model = Model(inputs = [Base_model.input], outputs = [w])

model.layers[-5].trainable = True
model.layers[-6].trainable = True
model.layers[-7].trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

batch_size = 8
n_epochs = 18

model.fit(X_train, y_train, batch_size = batch_size, epochs = n_epochs, 
          verbose = 1, validation_data = (X_val, y_val))

with open ("mobilenet_submission.csv", "w", encoding="ISO-8859-1", newline='') as fp:
    fp.write("Id,Category\n")
    i = 0
    for img_name in os.listdir(path_to_test):
        img_name = path_to_test + os.sep + img_name
        #print(img_name)
        image = Image.open(img_name)
        image = image.resize((224,224))
        image_array = np.asarray(image)
        image_array = image_array.astype(np.float16)
        image_array = image_array[np.newaxis, ...]
        predictions = model.predict(image_array)
        label = class_names[np.argmax(predictions)]
        fp.write("%d,%s\n" % (i, label))
        i+=1

                                                    