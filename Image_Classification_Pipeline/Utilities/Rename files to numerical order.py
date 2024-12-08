#rename files to numerical order

import os

# Set the directory path
directory = 'Akita'

# Set the starting number
num = 1

# Loop through files in the directory
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Construct the new filename with the numerical name
        new_filename = "{:03d}.{}".format(num, filename.split(".")[-1])
        num += 1

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))