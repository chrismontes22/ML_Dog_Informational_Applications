"""Used to remove the 'Dog' from each file name in a folder that has files ending with 'dog'"""
"""Removes everything after the space and readds .txt"""

import os
import shutil

def rename_files():
    directory = "your_directory"
    new_directory = "your_new_directory"
    os.makedirs(new_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith("Dog.txt"):
            new_filename = filename.rsplit(' ', 1)[0] + '.txt'
            shutil.copy(os.path.join(directory, filename), os.path.join(new_directory, new_filename))

rename_files()
