import sys
from glob import glob
import os
from collections import defaultdict
from random import shuffle, seed
from scipy.misc import imread, imresize, imsave

seed(101)  # this ensures that randomly generated numbers are kept the same on each run


def process_cats_dogs(raw_dir, processed_dir):
    """
    Function to split a folder of images by class (cats and dogs) into training and testing
    :param raw_dir: the raw images to process (cat file names are capitalized, dog file names aren't)
    :param processed_dir: output directory to store processed images
    """

    # Make output directory, if it doesn't exist already
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    #  Make train and test dirs
    train_dir = make_subdir(processed_dir, 'training')
    test_dir = make_subdir(processed_dir, 'testing')
    class_names = ['cat', 'dog']

    # Get list of all images
    all_images = glob(os.path.join(raw_dir, '*.jpg'))
    images_by_class = defaultdict(list)

    for img in all_images:

        if os.path.basename(img)[0].isupper():
            images_by_class['cat'].append(img)
        else:
            images_by_class['dog'].append(img)

    print 'Cats: ' + str(len(images_by_class['cat']))
    print 'Dogs: ' + str(len(images_by_class['dog']))

    # Process n images from each class with which to train and test
    n_train = 1000
    n_test = 200

    for class_name in class_names:

        # Make sub directories to store processed images
        train_out = make_subdir(train_dir, class_name)
        test_out = make_subdir(test_dir, class_name)

        # Randomly shuffle the data - this is performed "in place"
        shuffle(images_by_class[class_name])

        # We don't want our training data to "leak" into our testing!
        train_imgs = images_by_class[class_name][:n_train]
        test_imgs = images_by_class[class_name][n_train:n_test]

        # Process these images using a helper function
        process_imgs(train_imgs, train_out)
        process_imgs(test_imgs, test_out)


def process_imgs(img_list, out_dir, new_shape=(224, 224, 3)):
    """
    A function to read, resize and save an arbitrary list of images to disk (uses scipy.misc)
    :param img_list: list of images to be processed
    :param out_dir: folder where the images will be stored
    :param new_shape: the dimensions to which the images will be resized
    :return:
    """

    for img_file in img_list:

        img_arr = imread(img_file)
        img_resized = imresize(img_arr, new_shape)

        new_file = os.path.join(out_dir, os.path.basename(img_file))
        imsave(new_file, img_resized)


def make_subdir(parent_dir, sub_dir):
    """
    A function that creates a new subdirectory (if not already existing)
    :param parent_dir: the parent directory
    :param sub_dir: the name of the sub-directory
    :return: the path to the newly created directory
    """

    new_dir = os.path.join(parent_dir, sub_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir


if __name__ == '__main__':  # this is the "entry point" of program

    # sys.argv is the list of command line arguments (0 is the python script itself)
    process_cats_dogs(sys.argv[1], sys.argv[2])
