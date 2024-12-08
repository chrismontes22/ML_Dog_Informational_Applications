#This code is used to gather all of the labels by listing all of the folders in the Train directory

import os

# Replace 'your_main_folder_path' with the actual path to your main folder
main_folder_path = 'DD - Copy\\train'

# Get a list of all subdirectories (recursively)
subdirs = [x[0] for x in os.walk(main_folder_path)]

# Extract only the folder names (without full paths)
folder_names = [os.path.basename(subdir) for subdir in subdirs]

# Print the list of folder names
print(folder_names)
