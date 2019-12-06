# Path for the 3 needed files or directories

path_to_images = '/content/assignment_imgs/' # Directory with the 3000 images (unzipped)
path_to_annotations = '/content/drive/My Drive/img_annotations.json' # json annotations file
# Text file manually created from the excel table with all the bounding boxes IDs of aliments that may contain tomatoes
path_to_annotations_tomato = '/content/Annotations_with_tomato.txt'

model_choice = 'transfer' # (either 'scratch' or 'transfer')

img_dim = 350 # Dimension we want to resize the image to (image is 600*600 initially)
epochs = 20 # Number of epochs we want to train the model with 

# Packages 
import json
import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Add
from matplotlib import pyplot as plt

# Reading of annotation and annotation_tomato files

with open(path_to_annotations) as json_file:
    annotations = json.load(json_file)

f = open(path_to_annotations_tomato, 'r')
annotations_tomato = f.read()
annotations_tomato = annotations_tomato.split('\n')

# Creation of the labels dictionnary {image_ID : 0 or 1}
# 0 if the image does not have any tomato annotation
# 1 if the image contain at least 1 annoation with tomato

labels_dict =  {}
for i in range(len(annotations)): 
    labels_dict[list(annotations.items())[i][0]] = 0
    for bbox in list(annotations.items())[i][1]:
        if bbox['id'] in annotations_tomato:
            labels_dict[list(annotations.items())[i][0]] = 1

# Creation of the images and labels arrays

images = []
labels = []

for ID in os.listdir(path_to_images):
    images.append(cv2.imread(path_to_images+ID))
    labels.append(labels_dict[ID])

# Reshape of the 13 images that are not 600*600 ( 10 are 600*601, 1 is 600*654, 1 is 600*664, 1 is 986*600)
for i in range(len(images)):
    if images[i].shape != (img_dim, img_dim, 3):
        images[i] = cv2.resize(images[i],(img_dim, img_dim))

images = np.array(images, dtype="float") / 255.0  # not enough RAM with colab
labels = np.array(labels)

# One hot encoding of the labels
labels = pd.get_dummies(labels).values

# Data splitting into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Data augmentation to have more images to train the network
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,height_shift_range=0.2,
                         shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
datagen.fit(X_train)

# Hyperparameters values
batch_size = 64
learning_rate = 0.00001
opt = Adam(lr=learning_rate, decay=learning_rate / 10)

# Model trained from scratch
def model_scratch(input_shape):

    model = Sequential()

    model.add(Conv2D(32,  (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model

# Model with transfer learning
def model_transfer(X_train, X_test, y_train, y_test, path_classifier_weights = 'fc_model.h5'):

	model_vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

	train_data = model_vgg16.predict(X_train, len(X_train) // 64) 
	test_data = model_vgg16.predict(X_test, len(X_test) // 64)

	def classifier(input_shape_2):
		model = Sequential()
		model.add(Flatten(input_shape=input_shape_2))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(2, activation='softmax'))
		return model

	model_classifier = classifier(train_data.shape[1:])
	model_classifier.compile(optimizer=Adam(lr=0.00001, decay=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
	model_classifier.fit(train_data, y_train, epochs=50, batch_size=64,
				validation_data=(test_data, y_test))

	model_classifier.save_weights(path_classifier_weights)

	model = Sequential()
	model.add(VGG16(weights='imagenet', include_top=False, input_shape = X_train.shape[1:] ))
	top_model = classifier(model.output_shape[1:])
	top_model.load_weights(path_classifier_weights)
	model.add(top_model)
	for layer in model.layers[:15]:
		layer.trainable = False
	
	return model

# Choice of the model
if model_choice == 'scratch':
	model = model_scratch(X_train.shape[1:])
if model_choice == 'transfer':
	model = model_transfer(X_train, X_test, y_train, y_test,'fc_model.h5')

# Model compilation and checkpoint creation
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
callbacks = [ModelCheckpoint("/content/checkpoint.hdf5", monitor='val_acc',save_best_only=True)]

# Model Training
H = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
steps_per_epoch=len(X_train) // batch_size, validation_data=(X_test, y_test),
epochs=epochs, verbose=0, callbacks=callbacks)

# Plots of the training and test accuracy
plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('Model with transfer learning accuracy')
plt.xlim([0, 19])
plt.xticks(np.arange(1, 20, 2),np.arange(2, 21, 2))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='center left')
plt.savefig('Transfer_learning.png')
plt.show()

# has_tomatoes() prediction function
def has_tomatoes(image_path):
    im = cv2.imread(image_path)
    im = cv2.resize(im,(350, 350))
    IM = np.zeros((1,im.shape[0],im.shape[1],im.shape[2]))
    IM[0] = im
    pred = model.predict(IM)
    if pred[0,0] > pred[0,1]:
        return False
    else:
        return True
