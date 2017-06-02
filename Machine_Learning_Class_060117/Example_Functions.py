import os
import shutil
import glob

def create_new_directory(input_directory):

    """
    Creates a directory using the os library.

    Parameters
    ----------

    input_directory: string
        The folder to create.
    """
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)


def delete_directory(input_directory):

    """
    Deletes a directory using the shutil library.

    Parameters
    ----------

    input_directory: string
        The folder to delete.
    """

    shutil.rmtree(input_directory)

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

def grab_images(input_directory, image_search):

    """
    A function to return a list of images from a given directory,
    according to a given search phrase, image_search.

    Parameters
    ----------

    input_directory: string
        The folder to search.
    image_search: string
        A phrase that all found files must match. Can include '*', the "wildcard" character.
        Example: "*.jpg"

    Returns
    -------
    file_list: list
        A list of filenames in string format.

    """

    file_list = glob.glob(os.path.join(raw_dir, image_search))
    return file_list