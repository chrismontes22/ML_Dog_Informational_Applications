"""Added 'Dog' to the label. When it interacts with the LLM it wond be looking for Pitbull the Singer"""

# Define the path of the text file you want to modify
input_file = "input"

# Define the path of the new file where you want to save the modified content
output_file = "output"

# Define the text you want to replace and the new text
old_text = 'Pitbull”}'
new_text = 'Pitbull Dog”}'

# Open the input file in read mode and the output file in write mode
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    # Read the content of the input file
    content = infile.read()
    
    # Replace the old text with the new text
    modified_content = content.replace(old_text, new_text)
    
    # Write the modified content to the output file
    outfile.write(modified_content)

print(f"The text file {input_file} has been processed and the modified content has been saved as {output_file}.")
