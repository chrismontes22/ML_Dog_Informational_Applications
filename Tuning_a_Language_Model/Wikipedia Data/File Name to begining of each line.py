"""Takes the filename and adds it and a tab to the begining of each line in the text"""

import os

# Define the source and destination directories
input_dir = 'input_dir'
output_dir = 'output_dir'

# Create the destination directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each file in the source directory
for filename in os.listdir(input_dir):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Open the source file and the destination file
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as input_file, \
             open(os.path.join(output_dir, filename), "w", encoding="utf-8") as output_file:
            # Iterate over each line in the source file
            for line in input_file:
                # If the line is not blank, prepend the file name and a tab
                if line.strip():
                    line = filename.replace(".txt", "") + line
                # Write the line to the destination file
                output_file.write(line)
