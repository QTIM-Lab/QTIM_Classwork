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
    Parameters
    ----------

    raw_dir: string
        the raw images to process (cat file names are capitalized, dog file names aren't)
    processed_dir: string
        output directory to store processed images
    """

    # Make an output directory, if it doesn't exist already.
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    #  Make training and testing directories.
    train_dir = make_subdir(processed_dir, 'training')
    test_dir = make_subdir(processed_dir, 'testing')

    # Get a list of all images in the original data folder.
    all_images = glob(os.path.join(raw_dir, '*.jpg'))
    images_by_class = defaultdict(list)

    """
    Upper case images are cats. Lower case images are dogs. Why was this strange convention
    chosen? Who knows. But you will find many other strange ways to organize data found online,
    so dealing with this method is good practice.
    """
    for image in all_images:

        # Get the first letter of an image's filename.
        end_of_image_path = os.path.basename(image)
        first_letter = end_of_image_path[0]

        # If you want to see output as your images are counted, uncomment the line below:
        print end_of_image_path

        # Check if that letter "isupper" and sort accordingly.
        if first_letter.isupper():
            images_by_class['cat'].append(image)
        else:
            images_by_class['dog'].append(image)

    # Check our work by printing how many cats and dogs we found.
    print 'Number of Cats: ' + str(len(images_by_class['cat']))
    print 'Numer of Dogs: ' + str(len(images_by_class['dog']))

    # Set the number of images for training and testing in your neural network.
    number_of_training = 1000
    number_of_testing = 200

    # Create each class folder, and being to sort images.
    class_names = ['cat', 'dog']
    for class_name in class_names:

        # Make sub-directories in your training and testing directories for 'cat' and 'dog'.
        train_output = make_subdir(train_dir, class_name)
        test_output = make_subdir(test_dir, class_name)

        # Randomly shuffle the data. The data is currently organized by species type. We want a diversity of species for this task.
        shuffle(images_by_class[class_name])

        # We don't want our training data to "leak" into our testing! 
        # We use Python's list "slicing" mechanic to ensure we've correctly split the list in two.
        train_images = images_by_class[class_name][0:number_of_training]
        test_images = images_by_class[class_name][number_of_training:number_of_testing]

        # We're going to process these images using a helper function that we might want to use later,
        # rather than writing
        process_images(train_images, train_output)
        process_images(test_images, test_output)


def process_images(image_list, output_dir, new_shape=(224, 224, 3)):

    """
    A function to read, resize and save an arbitrary list of images to disk (uses the scipy.misc library)

    Parameters
    ----------
    image_list: list
        A list of images to be processed.
    output_dir: string
        A folder where the images will be saved to.
    new_shape: list or tuple
        The dimensions to which the images will be resized.
    """

    for image_file in image_list:

        # imread transfers an image file into a Python representation that we can then modify as we see fit.
        image_arr = imread(image_file)

        # Resize the image..
        image_resized = imresize(image_arr, new_shape)

        # Save it back out!
        new_file = os.path.join(output_dir, os.path.basename(image_file))
        imsave(new_file, image_resized)


def make_subdirectory(parent_dir, sub_dir):

    """
    A function that creates a new subdirectory (if not already existing) using os.

    Parameters
    ----------

    parent_dir: string
        the parent directory
    sub_dir: string
        the name of the sub-directory
    
    Returns
    -------
    new_dir: string
        the path to the newly created directory
    """

    new_dir = os.path.join(parent_dir, sub_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir


# This line is the "entry point" of the program. 
# This is what will run if you run this script from the command line as "python class_script.py"
if __name__ == '__main__':

    # sys.argv is the list of command line arguments (0 is the python script itself).
    # For example, python Process_Cats_and_Dogs.py ~/MyInputFolder ~/MyOutputFolder
    # sys.argv[0] = "Process_Cats_and_Dogs.py", sys.argv[1] = "~/MyInputFolder", sys.argv[2] = "~/MyOutputFolder"
    process_cats_dogs(sys.argv[1], sys.argv[2])