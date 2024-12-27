"""This script combines multiple text files into a single output file. Combines all the text from a single folder"""
#Just fill in the directory variables and run

import os

def combine_text_files_in_directory(input_directory, output_file):
    # Get a list of all text files in the input directory
    text_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in text_files:
            with open(os.path.join(input_directory, fname), encoding='utf-8') as infile:
                outfile.write(infile.read())
                  # add a newline between files

# Input directory
input_directory = 'input_directory'

# Output file
output_file = "output_file"

# Call the function to combine the text files in the directory
combine_text_files_in_directory(input_directory, output_file)

# Print a success message
print(f"The text files in {input_directory} were successfully combined into {output_file}.")
