""" This example will illustrate the process of image augmentation on a sample image from
    the Oxford flowers dataset to get a better feel for how image augmentation can effectively
    increase the size of your testing dataset.
"""

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# First, we're going to pick a sample file from the flowers dataset.
sample_file = '../data/oxfordflower17_organized/testing/0/image_0001.jpg'

# We'll then use keras' load_img command to store that file in an image class, and display it.
test_image = load_img(sample_file)
test_image.show()

# In order to work with the image as data, we'll need to transform it from keras' image class
# to a numpy array class. Additionally, we'll add a singleton dimension for ease of processing
# in the following steps.
test_array = img_to_array(test_image)  # this is a Numpy array with shape (3, 150, 150)
test_array = test_array.reshape((1,) + test_array.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

""" The magic of image augmentation comes with Keras' ImageDataGenerator function. It
    provides many options for modifiying data on the fly and in entirely random ways.
    This sample ImageDataGenerator uses most of the parameters available to modify our
    sample flower image. Change some of the parameters to see if it affects your output
    images. 
"""

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# How many copies of this image we want to create.
augmentation_multiplier = 40

""" The ImageDataGenerator's flow command will save randomly augmented images into a
    directory of our choosing with a prefix and format of our choosing.
"""

output_dir = '../Results/augmentation_test'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

num_images = 0
for batch in datagen.flow(test_array, batch_size=1, save_to_dir=output_dir, save_prefix='augmentation', save_format='jpeg'):
    num_images += 1
    if num_images > augmentation_multiplier:
        break  # otherwise the generator would loop indefinitely