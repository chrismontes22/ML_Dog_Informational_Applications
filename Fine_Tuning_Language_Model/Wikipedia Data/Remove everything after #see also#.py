#remove everything after see also in wikipedia data. Bad data lines for tuning LLM

import os

#insert your directories here
source_dir = "input_dir"
dest_dir = "output_dir"

# Create the destination directory if it does not exist
os.makedirs(dest_dir, exist_ok=True)

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file is a txt file
    if filename.endswith(".txt"):
        # Open the source file in read mode
        with open(os.path.join(source_dir, filename), "r", encoding="utf-8") as source_file:
            # Read the content of the source file
            content = source_file.read()
            # Find the position of '== See also =='
            pos = content.find("== See also ==")
            # If '== See also ==' is found, erase everything after it
            if pos != -1:
                content = content[:pos]
        # Open the destination file in write mode
        with open(os.path.join(dest_dir, filename), "w", encoding="utf-8") as dest_file:
            # Write the modified content to the destination file
            dest_file.write(content)

# Print a success message
print(f"All txt files in {source_dir} have been processed and the results have been saved in {dest_dir}.")
