""" The goal of this file is take a raw data folder, and transform into a
    file structure that is easy to use with the machine learning Python 
    library Keras.
"""

import glob
import os
import shutil

Unprocessed_Data_Directory = '../data/oxfordflower17'
Processed_Data_Directory = '../data/oxfordflower17_organized'

# We first check if our output directory exists. If not, we create it!
if not os.path.exists(Processed_Data_Directory):
    os.mkdir(Processed_Data_Directory)

""" The Oxford flower dataset is a dataset of 1360 images. There are 17 flower
    species with 80 examples of each species. Images 0001-0080, 0081-0160, etc.
    each belong to a different flower class. To get them ready for Keras, we
    need to make two folders: "training" and "testing", each with 17 subfolders
    for each class of flowers. We will train our network on the training folder
    and test it on the testing folder.
"""

# We first get a list of all of our flower image files using the python library glob
file_list = glob.glob(os.path.join(Unprocessed_Data_Directory, '*.jpg'))

# We sort them alphanemurically to make sure we can easily split them into classes.
file_list = sorted(file_list)

# We sort all of the filenames into a Python dictionary for organization's sake.
class_dictionary = {}
for class_num in xrange(17):
    class_dictionary[class_num] = file_list[(class_num*80):((class_num+1)*80)]

# We decide on a percentage to set aside for testing data, and create testing/training folders.
testing_ratio = .2

# Create training and testing folders
if not os.path.exists(os.path.join(Processed_Data_Directory, 'training')):
    os.mkdir(os.path.join(Processed_Data_Directory, 'training'))
if not os.path.exists(os.path.join(Processed_Data_Directory, 'testing')):
    os.mkdir(os.path.join(Processed_Data_Directory, 'testing'))


for class_num in class_dictionary:

    if not os.path.exists(os.path.join(Processed_Data_Directory, 'testing', str(class_num))):
        os.mkdir(os.path.join(Processed_Data_Directory, 'testing', str(class_num)))

    for image in class_dictionary[class_num][0:int(80*testing_ratio)]:
        shutil.copy(image, os.path.join(Processed_Data_Directory, 'testing', str(class_num), os.path.basename(image)))

    if not os.path.exists(os.path.join(Processed_Data_Directory, 'training', str(class_num))):
        os.mkdir(os.path.join(Processed_Data_Directory, 'training', str(class_num)))

    for image in class_dictionary[class_num][int(80*testing_ratio):]:
        shutil.copy(image, os.path.join(Processed_Data_Directory, 'training', str(class_num), os.path.basename(image)))      





