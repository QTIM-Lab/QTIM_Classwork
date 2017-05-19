from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

import numpy as np

# path to the model weights files.
weights_path = '../data/vgg16_weights.h5'
output_weights_path = '../data/bottleneck_weights.h5'

# Locate data to train and test on.
train_data_dir = '../data/oxfordflower17_organized/training'
test_data_dir = '../data/oxfordflower17_organized/testing'

# Instantiate some parameters for running the models later.
train_samples = 1088
test_samples = 272
epochs = 50
batch_size = 16
img_width, img_height = 150, 150

""" Load the VGG16 network with weights from imagenet. The VGG16
    network is a popular and successful model for objection detection
    in images. Keras comes with this network pre-loaded AND comes with
    weights that have proven successful on ImageNet, a large database
    of images used for object detection competitions.
"""
VGG_Model = applications.VGG16(include_top=False, weights='imagenet')

# Set up our data generator to automatically pull training data from our
# training directory. All images are rescaled to the intensity range 0-1.
# datagen = ImageDataGenerator(rescale=1. / 255)
# generator = datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=False)

# # Extract features from the penultimate layers of the VGG network
# # on our training images. This may take some time.
# bottleneck_features_train = VGG_Model.predict_generator(
#     generator, train_samples // batch_size, verbose=1)
# np.save(open('../Results/bottleneck_features_train.npy', 'w'),
#             bottleneck_features_train)

# # Repeat the bottle-necking process for testing data.
# generator = datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=False)
# bottleneck_features_test = VGG_Model.predict_generator(
#     generator, test_samples // batch_size, verbose=1)
# np.save(open('../Results/bottleneck_features_test.npy', 'w'),
#             bottleneck_features_test)

# Alternatively, we can load our saved arrays from a previous run of
# of the program.
bottleneck_features_train = np.load('../Results/bottleneck_features_train.npy')

print bottleneck_features_train.shape


test_data = np.load('../Results/bottleneck_features_test.npy')

# print bottleneck_features_test.shape

# fd = df

""" We will have to generate our own labels this time. Luckily, all of our data
    has the same amount of classes and is in order. We convert them to numpy arrays
    for better compatability with keras.
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
          validation_data=(test_data, train_labels))
model.save_weights(output_weights_path)
