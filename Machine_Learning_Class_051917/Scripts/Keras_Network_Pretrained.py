""" We take a different method of attack on the Oxford 17 Flowers
    dataset. This time, we are going to use an already-successful
    and widely-used network, the VGG16 network, as a starting point
    for our own classification problem. 
    
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

import numpy as np


# Locate data to train and test on.
train_data_dir = '../data/oxfordflower17_organized/training'
test_data_dir = '../data/oxfordflower17_organized/testing'

# Instantiate some parameters for running the models later.
train_samples = 1088
test_samples = 272
epochs = 20
batch_size = 16
img_width, img_height = 150, 150

# We load our saved arrays generated from Bottleneck_Data.py
bottleneck_features_train = np.load('../Results/bottleneck_features_train.npy')
bottleneck_features_test = np.load('../Results/bottleneck_features_test.npy')

""" We will have to generate our own labels this time. Luckily, all of our data
    has the same amount of classes and is in order. We convert them to numpy arrays
    for better compatability with keras using the "to_categorical" command. This converts
    a class value in the format [5] to [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0], which is more
    amenable to neural network calculations.
"""
train_labels = to_categorical(np.array([int(index/64) for index in range(64*17)]))
test_labels = to_categorical(np.array([int(index/16) for index in range(16*17)]))

""" We create a much simpler model than we did before. We can still find success with
    this model because the pre-calibrated VGG-net has already distilled a set of
    informative features from the original image to give us a "head-start". The hope is that these features will
    make our classification scheme go much faster.
"""
model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(bottleneck_features_train, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(bottleneck_features_test, test_labels))

model.save_weights('../Results/Pretrained_Network_Weights.h5')  # always save your weights after training or during training
