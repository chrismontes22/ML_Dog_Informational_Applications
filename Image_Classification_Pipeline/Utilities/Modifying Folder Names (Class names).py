#modifying folder nmaes (Class names) so they match the names on the text data

import os
import shutil

def rename_folders(root_dir, old_name, new_name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == old_name:
            new_dirpath = os.path.join(os.path.dirname(dirpath), new_name)
            shutil.move(dirpath, new_dirpath)
            print(f'Renamed directory {dirpath} to {new_dirpath}')

# Usage
root_dir = '.'  # replace with your directory path
rename_folders(root_dir, 'Japanese Spaniel', 'Japanes-Chin')
