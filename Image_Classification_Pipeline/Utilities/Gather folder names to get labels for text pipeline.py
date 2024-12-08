#Gather folder names to get labels for text pipeline

import os

def get_folder_names(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

# Replace 'your_directory_path' with the path of your directory
folder_list = get_folder_names('DD\\test')

num_folders = len(folder_list)
print(f"There are {num_folders} folders in the directory.")

print(folder_list)
