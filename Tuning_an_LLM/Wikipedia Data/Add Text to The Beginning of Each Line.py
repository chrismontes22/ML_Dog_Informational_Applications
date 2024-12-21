"""Add text to the begining of each line. goes through all of the txt files in a folder. Useful for json formatting"""

import os

# Define the path to the input and output directories
input_dir = 'input_dir'
output_dir = 'output_dir'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the symbols to be added
symbols = '{"text": "'

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        # Open the input file in read mode
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as infile:
            # Open the output file in write mode
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as outfile:
                # Iterate over each line in the input file
                for line in infile:
                    # Write the line with the symbols added to the output file
                    outfile.write(symbols + line)

print("Done!")
