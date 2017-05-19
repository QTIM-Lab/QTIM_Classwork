""" Load the VGG16 network with weights from imagenet. The VGG16
    network is a popular and successful model for objection detection
    in images. Keras comes with this network pre-loaded AND comes with
    weights that have proven successful on ImageNet, a large database
    of images used for object detection competitions.
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

VGG_Model = applications.VGG16(include_top=False, weights='imagenet')

# Set up our data generator to automatically pull training data from our
# training directory. All images are rescaled to the intensity range 0-1.
datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

# Extract features from the penultimate layers of the VGG network
# on our training images. This may take some time.
bottleneck_features_train = VGG_Model.predict_generator(
    generator, train_samples // batch_size, verbose=1)

# Save results for usage in the pretrained network.
np.save(open('../Results/bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

# Repeat the bottle-necking process for testing data.
generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_test = VGG_Model.predict_generator(
    generator, test_samples // batch_size, verbose=1)
np.save(open('../Results/bottleneck_features_test.npy', 'w'),
            bottleneck_features_test)