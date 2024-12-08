
"""This code is a modified version of my Text_Data_Pipeline.py that was use to streamline the fine tuning of an LLM.
Instead of using Bert for data augmentation and turning the file into JSON, after downloading and parsing the text,
it chunks the data with Langchain and organizes it by dog breed"""

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import os
import time
import shutil

######################### Recomended Script Modifiers####################################

#If you have the full list text file (Dog_List.txt), save it to the main folder and set FULL_LIST to True to doqnload all of the data for all available breeds
FULL_LIST = False
if FULL_LIST:
    def read_file_to_list(file_path):
        with open(file_path, 'r') as file:
            items_list = [line.strip() for line in file]
        return items_list
    file_path = 'Dog_List.txt'
    dog_list = read_file_to_list(file_path)
    print(dog_list)

#Insert your manual dog list here in the dog_breed variable, or pass the dog_list variable from the previous cell as: dog_breed = dog_list
"""The Dog_List.txt file is set to work with how the websites standardize the breed name, as well as for the dictionary below to manually gather the other breeds.
It is highly recomended to use the breed name provided in that list """
dog_breed = ["Saint-Bernard", "Rottweiler"]
dog_breed = [breed.title() for breed in dog_breed] #This makes the first letter of every word in the list Capitalized to match the dictionary casing (if needed)
directory_path= 'Dogs' #Path to save all of the files

#Feel free to mess with the chunk size and overlap HERE
chunk_size = 500
chunk_overlap = 120

#List of websites to gather data. You can turn them on or off here by setting to True or False.
DOGTIME = True
DAILYPAWS = True
CANINEJOURNAL = True
PETS4HOMES = True

###############################################################################################

"""Most websites have the dog breed standardized. These dictionaries are here (one for each site) for when some dog breeds are not standardized.
In that case, you would have to figure out where the URL error is and fix it in the dictionary in the format below by comparing the variable in the list to the actual URL.
Luckily, the script logs all the URL errors that occur, so you can let the script run once and then check out the URL log.
You can add your own here if there are other breeds outside of the Dog_List.txt that dont match the website"""

dogtime_W = {"German-Shepherd": "German-Shepherd-Dog", "Xoloitzcuintli": "Xoloitzuintli"}
dailypaws_W = {"Xoloitzcuintli": "Xoloitzcuintli-Mexican-Hairless", "Poodle": "Standard-Poodle"}
caninejournal_W = {"Chinese-Shar-Pei": "Shar-Pei"}
pets4homes_W = {
    "Chinese-Shar-Pei": "Shar-Pei",
    "Shiba-Inu": "Japanese-Shiba-Inu",
    "Belgian-Malinois": "Belgian-Shepherd-Dog",
    "Belgian-Sheepdog": "Belgian-Shepherd-Dog",
    "Bull-Terrier": "English-Bull-Terrier",
    "Bulldog": "English-Bulldog",
    "Collie": "Rough-Collie",
    "Doberman-Pinscher": "Dobermann",
    "Great-Pyrenees": "Pyrenean-Mountain-Dog",
    "Pembroke-Welsh-Corgi": "Welsh-Corgi-Pembroke",
    "Vizsla": "Hungarian-Vizsla",
    "Xoloitzcuintli": "Mexican-Hairless"
}


#Parse the html and output a text file
def fetch_and_parse(url, dog_breed, wnum):
    response = requests.get(url)

    if response.status_code != 200:
        print(f'Failed to retrieve {url}. Status code: {response.status_code}')
        with open('failed_urls.txt', 'a') as f:
            f.write(f'{url}\n')  # Write the failed URL and a newline character
        return []  # Return an empty list so the script continues

    soup = BeautifulSoup(response.text, 'html.parser') # Parse the HTML content with BeautifulSoup.
    texts = [p.get_text() for p in soup.find_all('p')]   # Find all <p> tags and extract the text

    with open(f'{dog_breed}{wnum}.txt', 'w', encoding = 'utf-8') as f: # Write the texts to a file
        f.write('\n'.join(texts))
    return texts if texts else []

#This process the file further, chunks it with Langchain (recursive character)
def process_and_chunk_text(texts, file_name, x=chunk_size, y=chunk_overlap):
    texts = [line.strip() for line in texts if len(line.strip()) > 55]
    time.sleep(2)

    # Combine writing and chunking to minimize file operations
    text_to_write = '\n'.join(texts)

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=x,
        chunk_overlap=y
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text_to_write)

    # Write the chunks back to the file
    with open(file_name, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n")

    if os.path.getsize(file_name) == 0:
        os.remove(file_name)

##################################################################################
"""Here is the section for the different websites. You will notice that the website functions have similar structures.
Some minor code adjustments is usually enough when adding a new website to go from HTML to JSON training data.
Make sure to add the line for wnum, this is to identify the text file to the corresponding website"""

def dogtime_data(dog_breed):
    wnum = 1 #This helps name the files that get outputed, such as file1.txt, file1.json
    file_name = f'{dog_breed}{wnum}.txt'
    used_data_file_path = f"{directory_path}/used_data/{dog_breed}{wnum}.txt"
    if not os.path.exists(used_data_file_path):
        dogtime_name = dogtime_W.get(dog_breed, dog_breed).lower() #Retrieves the unique breed name from the site if it isn't like the list. Some sites require all lower cased so this handles with ".lower"
        url = f'https://dogtime.com/dog-breeds/{dogtime_name}' #URL name and variable
        texts = fetch_and_parse(url, dog_breed, wnum)

        # Remove unwanted lines
        start_remove = 'Looking for the best dog for your apartment?'
        end_remove = 'Playing with our pups is good for us.'

        start_index = None
        end_index = None

        for i, line in enumerate(texts):
            if line.startswith(start_remove):
                start_index = i
            elif line.startswith(end_remove):
                end_index = i
                break

        if start_index is not None and end_index is not None:
            texts = texts[:start_index] + texts[end_index+1:]
        if texts:
            texts = texts[:-2]
        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

def dailypaws_data(dog_breed):
    wnum = 2
    file_name = f'{dog_breed}{wnum}.txt'
    used_data_file_path = f"{directory_path}/used_data/{dog_breed}{wnum}.txt"
    if not os.path.exists(used_data_file_path):
        dailypaws_name = dailypaws_W.get(dog_breed, dog_breed).lower()
        url = f'https://www.dailypaws.com/dogs-puppies/dog-breeds/{dailypaws_name}'
        texts = fetch_and_parse(url, dog_breed, wnum)

        # Omit the second, third, and fourth lines from the texts
        del texts[1:4]

        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

def caninejournal_data(dog_breed):
    wnum = 3
    file_name = f'{dog_breed}{wnum}.txt'
    used_data_file_path = f"{directory_path}/used_data/{dog_breed}{wnum}.txt"
    if not os.path.exists(used_data_file_path):
        caninejournal_name = caninejournal_W.get(dog_breed, dog_breed).lower()
        url = f"https://www.caninejournal.com/{caninejournal_name}"
        texts = fetch_and_parse(url, dog_breed, wnum)
        # Remove the first and last 5 lines
        texts = texts[5:-5]
        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

def pets4homes_data(dog_breed):
    wnum = 4
    file_name = f'{dog_breed}{wnum}.txt'
    used_data_file_path = f"{directory_path}/used_data/{dog_breed}{wnum}.txt"
    if not os.path.exists(used_data_file_path):
        pets4homes_name = pets4homes_W.get(dog_breed, dog_breed).lower()
        url = f"https://www.pets4homes.co.uk/dog-breeds/{pets4homes_name}"
        texts = fetch_and_parse(url, dog_breed, wnum)
        # Remove the first and last 5 lines
        texts = texts[5:-5]
        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

##################################################################################
#Moves all of the text files into a subfolder to clean up the work folder
def move_text_files(folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Get a list of all text files in the current directory
    text_files = [f for f in os.listdir() if f.endswith('.txt')]

    # Exclude specific files from being moved
    excluded_files = ['failed_urls.txt', 'failed_json.txt']
    text_files = [f for f in text_files if f not in excluded_files]

    # Move each text file into the folder
    for file in text_files:
        destination_file = os.path.join(folder_name, file)
        if os.path.exists(destination_file):
            os.remove(destination_file)  # Remove the file if it already exists
        shutil.move(file, folder_name)

#Function to merge all files with the same base name into one. It saves the individual files under "used_data"
def merge_text_files(directory, dog_breed):
    # Ensure the 'used data' folder exists
    used_data_dir = os.path.join(directory, 'used_data')
    if not os.path.exists(used_data_dir):
        os.makedirs(used_data_dir)

    # Merge text files by breed
    parent_file = os.path.join(directory, f'{dog_breed}.txt')

    for filename in os.listdir(directory):
        if filename.startswith(dog_breed) and filename != f'{dog_breed}.txt':
            child_file = os.path.join(directory, filename)

            # Append the content of child files to the parent file
            with open(child_file, 'r', encoding='utf-8') as f_child, \
                 open(parent_file, 'a', encoding='utf-8') as f_parent:
                f_parent.write(f_child.read() + '\n')

            # Move the child file to 'used data' folder
            shutil.move(child_file, used_data_dir)

    with open(parent_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(parent_file, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.strip():
                file.write(line)

directory_path= 'Dogs'
#Now call the website functions in a loop for the list
for breed in dog_breed:
    if DOGTIME:
        dogtime_data(breed)
    if DAILYPAWS:
        dailypaws_data(breed)
    if CANINEJOURNAL:
        caninejournal_data(breed)
    if PETS4HOMES:
        pets4homes_data(breed)
    move_text_files("Dogs")
    merge_text_files(directory_path, breed)

