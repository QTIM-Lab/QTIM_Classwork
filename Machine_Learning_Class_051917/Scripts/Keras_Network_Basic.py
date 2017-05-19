""" Our goal in this piece of code is to run a model that will attempt to learn the differences
    between 17 different species of flowers from a dataset of 1360 images. We'll be using a 
    deep convolutional neural network implemented by the library Keras, which wraps around
    the machine learning library Theano.
"""

import os

# We import only the functions we need from Keras. One can always import the full library when exploring.
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

""" Our first step is to create a neural network "model" using the keras
    framework. This will be a sequential model -- i.e. a model where inputs
    follow in a logical, first-this-then-that series of steps. The particular
    sort of neural networks we will be using are called convolutional neural networks.
    The network is three layers deep. At each layer, a CNN is trained on the data,
    passed through an activation layer, and then reduced in size by the MaxPooling
    layer. Finally, the multi-dimensional feature maps produced by the CNN is
    flattened into a one-dimensional feature vector. This feature vector is passed
    through a dense layer - i.e. a non-CNN layer meant for one-dimensional feature
    vectors. Finally, 50% of connections between nodes are dropped-out, i.e. set to
    zero. This step prevents over-fitting by de-stressing the importance of any one
    connection. Finally, a dense layer reduces us down to our 17 flower classes, and
    a sigmoid layer prepares it to be used for maximum likelihood classification during training.
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(90, 90, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(17))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

# This is the augmentation configuration we will use for training.
# It shears, zooms, and horizontally flips input images. Additionally
# it rescales image intensities to the range [0 ... 1]
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# The testing set will have no augmentation. We only rescale the
# inpute values from 0 to 1 instead of from 1 to 255.
test_datagen = ImageDataGenerator(rescale=1./255)

# This is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../data/oxfordflower17_organized/training',  # this is the target directory
        target_size=(90, 90),  # all images will be resized to 90x90
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../data/oxfordflower17_organized/testing',
        target_size=(90, 90),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=640 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

model.save_weights('first_try.h5')  # always save your weights after training or during training