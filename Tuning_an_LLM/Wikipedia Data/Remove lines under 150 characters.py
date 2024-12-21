"""When the data was split into lines, some of the lines simply had a name or only partial sentences.
This erases any line under the linelen value"""

#Within a folder full of text files, modify this variable to choose how long the shortest line whould be
linelen = 150

def filter_lines(txt_path, new_txt_path):
    # Read the content of the txt file
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    # Filter out lines that are less than 150 characters long
    lines = [line for line in lines if len(line) >= linelen]

    # Write the filtered lines to a new txt file
    with open(new_txt_path, 'w', encoding='utf-8') as new_txt_file:
        new_txt_file.writelines(lines)

    # Find the length of the shortest line in the filtered data
    shortest_line_length = min(len(line) for line in lines)

    return shortest_line_length

# Call the function to filter the lines in "my_data.txt" and save it as "filtered_data.txt"
shortest_line_length = filter_lines("new_dog.txt", "dog3.txt")

print(f"The shortest line in the filtered data has {shortest_line_length} characters.")
