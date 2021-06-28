#from classification_models.tfkeras import Classifiers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import metrics
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import cv2
import os

batch_size = 64
epochs = 15

#Create an index of class names
path = os.getcwd()
path_to_train = path+os.sep+'vehicle'+os.sep+'train'+os.sep+'train'
path_to_test = path+os.sep+'vehicle'+os.sep+'test'+os.sep+'testset'

class_names = sorted(os.listdir(path_to_train))

# Find all image files in the data directory.
X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
for root, dirs, files in os.walk(path_to_train):
    for name in files:
        if name.endswith(".jpg"):
            # Load the image:
            img_name = root + os.sep + name
            print(img_name)
            
            img = plt.imread(img_name)
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32)
            img -= 128

            
            # Add images to  the list
            X.append(img)
            # Extract class name from the directory name:
            label = img_name.split(os.sep, -2)
            y.append(class_names.index(label[-2]))

# Cast the python lists to a numpy array.
data = np.array(X)
labels = np.array(y)
# convert label to one-hot vector
y_one_hot = to_categorical(y)

# Split the train and test
X_train, X_test, y_train, y_test = train_test_split(data, y_one_hot, test_size = 0.2)


#Prepare a pretrained CNN for feature extraction
base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                            input_shape=(224, 224, 3),
                                                            weights='imagenet')
#base_model.trainable = False

in_tensor = base_model.inputs[0] # Grab the input of base model
out_tensor = base_model.outputs[0] # Grab the output of base model
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)
#out_tensor = Dense(256, activation='relu')(out_tensor)
#out_tensor = Dropout(0.4)(out_tensor)
out_tensor = Flatten()(out_tensor)
out_tensor = Dense(256, activation='relu')(out_tensor)
out_tensor = Dropout(0.4)(out_tensor)
out_tensor = Dense(17, activation='softmax')(out_tensor)
# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs = [in_tensor],
                              outputs = [out_tensor])

## augmentation cho training data
#aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
#                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
#
## augementation cho test
#aug_test= ImageDataGenerator(rescale=1./255)

opt = optimizers.Adam(learning_rate=0.00001)

# Compile the model for execution. Losses and optimizers
model.compile(loss = 'categorical_crossentropy', 
              optimizer = opt,
              metrics=['accuracy'])
model.summary()

for layer in base_model.layers:
    layer.trainable = False

weighted = ModelCheckpoint('weighted.hdf5', save_best_only=True, save_weights_only=True);

#model.fit_generator(aug_train.flow(X_train, y_train, batch_size=batch_size),
#          steps_per_epoch=len(X_train)//batch_size,
#          validation_steps=len(X_test)//batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=aug_test.flow(X_test, y_test, batch_size=batch_size),
#          callbacks=[weighted])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[weighted])

#model.load_weights('weighted.hdf5')


X_test = []

for img_name in os.listdir(path_to_test):
        # Load the image
        img_name = path_to_test+os.sep+img_name 
        print(img_name)
        img_test = plt.imread(img_name)
        # Resize the image to fit the net input size
        img_test = cv2.resize(img_test, (224, 224))
                
        # Convert data to float, and remove mean
        img_test = img_test.astype(np.float32)
        img_test -= 128
        
        X_test.append(img_test)
#        X_test.append(img_flip)

X_test = np.array(X_test)

y_test_pred = model.predict(X_test)

with open("inception_submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    for i in range(len(y_test_pred)):
        # Convert class id to name and write to file
        label = class_names[np.argmax(y_test_pred[i])]
#        img_name = img_name.split(os.sep)[-1]
#        img_name = img_name.split('.')[0]
        fp.write("%i,%s\n" % (i, label))