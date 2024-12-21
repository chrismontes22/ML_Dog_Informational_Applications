"""Goes through all of the text files in a folder and replaces a strign of text with another text.
I used it to replace some negative terms, but also to format into JSON"""
#Just insert the variables folder directories of where you want to load/save, and the variables of the old and new words

import os
import shutil

# Define the directory containing the text files
input_dir = input_dir

# Define the directory to save the modified text files
output_dir = output_dir

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the word to be replaced and the new word
old_word = '{"text": "'
new_word = ''

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        # Open the text file in read mode
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
            # Read the content of the file
            file_content = file.read()
            
            # Replace the old word with the new word
            modified_content = file_content.replace(old_word, new_word)
        
        # Open the text file in write mode in the output directory
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as file:
            # Write the modified content to the file
            file.write(modified_content)

print(f"All text files in {input_dir} have been processed and the modified files have been saved in {output_dir}.")
