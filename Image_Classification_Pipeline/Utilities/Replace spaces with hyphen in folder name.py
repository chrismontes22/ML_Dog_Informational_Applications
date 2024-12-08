#Replace spaces with hyphen in folder (class) names so it automates into the text pipline

import os

# specify your directory
your_directory = 'DC6'

# iterate over all the directories in the specified directory
for dirpath, dirnames, filenames in os.walk(your_directory):
    # iterate over each directory name
    for dirname in dirnames:
        # create the original directory path
        original_dir_path = os.path.join(dirpath, dirname)
        # create the new directory path by replacing spaces with hyphens
        new_dir_path = os.path.join(dirpath, dirname.replace(' ', '-'))
        # rename the directory
        os.rename(original_dir_path, new_dir_path)
