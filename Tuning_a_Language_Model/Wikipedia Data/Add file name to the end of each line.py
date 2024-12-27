"""Adds the filename (dog breed) to the end of each line"""

import os

# Define the path to the input and output directories
input_dir = 'input_dir'
output_dir = 'output_dir'
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        # Get the base name of the file (without the extension)
        basename = os.path.splitext(filename)[0]
        # Open the input file in read mode
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as infile:
            # Open the output file in write mode
            with open(os.path.join(output_dir, basename + '.txt'), 'w', encoding='utf-8') as outfile:
                # Iterate over each line in the input file
                for line in infile:
                    # Write the line with the base name added to the end of the line in the output file
                    outfile.write(line.rstrip('\n') + ' ' + basename + '\n')

print("Done!")
