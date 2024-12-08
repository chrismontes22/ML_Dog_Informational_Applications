
"""Here is the code for downloading, cleaning, formatting/transforming to JSON, and uploading the data. Options to use Mask/Unmask (DistillBERT) 
for data augmentation."""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from transformers import pipeline
import random
import shutil

#Insert your list here in the dog_breed variable
dog_breed = ['Rottweiler', 'Saint-Bernard', 'Shiba-Inu']

"""Most websites have the dog breed standardized. These dictionaries are here (one for each site) for when some dog breeds are not standardized.
In that case, you would have to figure out where the URL error is and fix it in the dictionary in the format below by comparing the variable in the list to the actual URL.
Luckily, the script logs all the URL errors that occur, so you can let the script run once and then check out the URL log."""

dogtime_W = {}
dailypaws_W = {}
caninejournal_W = {}
pets4homes_W = {"Shiba-Inu":"Japanese-Shiba-Inu"}

final_json_name = 'Name_JSON_TO_UPLOAD' #The name of the Final Json file that you want.

#List of websites. You can turn them on or off here by setting to True or False.
DOGTIME = False
DAILYPAWS = False
CANINEJOURNAL = True
PETS4HOMES = True

USE_MASKING = True #Data Augmentation by masking then unmasking with the BERT Transformers. Adds about 30 seconds per class. For further augmentation use the backtranslator in this repository.
UPLOAD_TO_HF = True #Upload to huggingface

#Parse the html and output a text file
def fetch_and_parse(url, dog_breed, wnum):
    response = requests.get(url)
     
    if response.status_code != 200:
        print(f'Failed to retrieve {url}. Status code: {response.status_code}')
        with open('failed_urls.txt', 'a') as f: 
            f.write(f'{url}\n')  # Write the failed URL and a newline character
        return []  # Return an empty list so the script continues
    
    if not USE_MASKING:
        time.sleep(5) # Slows down the script so that it doesn't make too many requests to the site at once. Not necessary with Masking because it already slows the script.
    
    soup = BeautifulSoup(response.text, 'html.parser') # Parse the HTML content with BeautifulSoup.
    texts = [p.get_text() for p in soup.find_all('p')]   # Find all <p> tags and extract the text
    
    with open(f'{dog_breed}{wnum}.txt', 'w', encoding = 'utf-8') as f: # Write the texts to a file
        f.write('\n'.join(texts))
    return texts if texts else []


#Common formatting needed by each of the website text to turn into JSON
def format_and_write_to_json(dog_breed, texts, wnum):
    if not texts:
        return  # Exit the function if there are no texts to process
    try:
        texts = [line for line in texts if len(line) > 55] # Filter out lines that are less than 50 characters
        texts = [{"text": f"{dog_breed}: {line}", "label": f"Please tell me something interesting about the {dog_breed} Dog"} for line in texts] # Add the required strings to each line
        json_text = json.dumps(texts, ensure_ascii=False, indent=4) # Serialize the texts list to a JSON string
        
        with open(f'{dog_breed}{wnum}.json', 'w', encoding='utf-8') as f:  # Write the JSON string to the file
            f.write(json_text)

    except (ValueError, TypeError) as e:
        with open('failed_json.txt', 'a') as f:
            f.write(f'Failed to format JSON for {dog_breed}{wnum}.json due to: {str(e)}\n')
    except:
        pass

# Initialize the fill-mask pipeline with distilbert. This goes outside the mask_and_unmask function since we only need to call this once
if USE_MASKING:
    fill_mask = pipeline(
        "fill-mask",
        model="distilbert-base-cased",  #The model and tokenizer for masking/unmasking.
        tokenizer="distilbert-base-cased" #Make sure to use "cased" and not "uncased" or else everything will be lower cased.
    )


def mask_and_unmask(dog_breed, wnum, n=4):
    def apply_mask_unmask(text):
        # Apply n masks to the text
        for _ in range(n):
            words = text.split()
            mask_positions = [i for i, word in enumerate(words) if word.isalpha()]
            if not mask_positions:
                break
            mask_pos = random.choice(mask_positions)
            words[mask_pos] = fill_mask.tokenizer.mask_token
            text = " ".join(words)
            
            # Unmask
            predictions = fill_mask(text)
            text = predictions[0]['sequence']
        return text

    try:
        # Load the data from the input JSON file
        with open(f'{dog_breed}{wnum}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Apply the mask and unmask operation to each "text" field
        for item in data:
            item_dog_breed, text = item['text'].split(':', 1)
            # Check if the dog breed matches the target breed
            if item_dog_breed.strip().lower() == dog_breed.lower():
                text = apply_mask_unmask(text.strip())
                item['text'] = f"{item_dog_breed}: {text}"

        # Write the augmented data to a new output JSON file
        with open(f'{dog_breed}_unmasked{wnum}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

##################################################################################
"""Here is the section for the different websites. You will notice that the website functions have similar structures.
Some minor code adjustments is usually enough when adding a new website to go from HTML to JSON training data"""

def dogtime_data(dog_breed):
    wnum = 1 #This helps name the files that get outputed, such as file1.txt, file1.json
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

    format_and_write_to_json(dog_breed, texts, wnum)
    if USE_MASKING:
        mask_and_unmask(dog_breed, wnum, n=4)

def dailypaws_data(dog_breed):
    wnum = 2
    dailypaws_name = dailypaws_W.get(dog_breed, dog_breed).lower()
    url = f'https://www.dailypaws.com/dogs-puppies/dog-breeds/{dailypaws_name}'
    texts = fetch_and_parse(url, dog_breed, wnum)
    texts = [line.lstrip() for line in texts]
    # Omit the second, third, and fourth lines from the texts
    del texts[1:4]
    format_and_write_to_json(dog_breed, texts, wnum)
    if USE_MASKING:
        mask_and_unmask(dog_breed, wnum, n=4)

def caninejournal_data(dog_breed):
    wnum = 3 
    caninejournal_name = caninejournal_W.get(dog_breed, dog_breed).lower()
    url = f"https://www.caninejournal.com/{caninejournal_name}"
    texts = fetch_and_parse(url, dog_breed, wnum)
    # Remove the first and last 5 lines
    texts = texts[5:-5]
    # Call the new function to process the text and write it to a JSON file
    format_and_write_to_json(dog_breed, texts, wnum)
    if USE_MASKING:
        mask_and_unmask(dog_breed, wnum, n=4)

def pets4homes_data(dog_breed):
    wnum = 4 
    pets4homes_name = pets4homes_W.get(dog_breed, dog_breed).lower()
    url = f"https://www.pets4homes.co.uk/dog-breeds/{pets4homes_name}"
    texts = fetch_and_parse(url, dog_breed, wnum)
    # Remove the first and last 5 lines
    texts = texts[5:-5]
    # Call the new function to process the text and write it to a JSON file
    format_and_write_to_json(dog_breed, texts, wnum)
    if USE_MASKING:
        mask_and_unmask(dog_breed, wnum, n=4)

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

#Merge all of the json files into one to organize the data better. Also moves all but the merged JSON to a subfolder
def merge_json_files(folder_path, output_file):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data.append(json.load(f))

    merged_data = sum(data, [])  
    cleaned_data = [merged_data[0]] + merged_data[1:-1] + [merged_data[-1]]  

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)  

    subfolder_name = "Used JSON Files"
    subfolder_path = os.path.join(folder_path, subfolder_name)

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename != output_file:
            file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(subfolder_path, filename)
            os.replace(file_path, new_file_path)

#Function to move the final json file into a folder
def move_completed_json(file_path, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Get the filename and folder path
    filename = os.path.basename(file_path)
    folder_path = os.path.join(os.getcwd(), folder_name)

    # Check if the file already exists in the folder
    if os.path.exists(os.path.join(folder_path, filename)):
        # Rename the file by adding a number in parentheses
        base, extension = filename.split('.')
        new_filename = f"{base} (1).{extension}"
        i = 1
        while os.path.exists(os.path.join(folder_path, new_filename)):
            i += 1
            new_filename = f"{base} ({i}).{extension}"
        filename = new_filename

    # Move the file to the folder
    shutil.move(file_path, os.path.join(folder_path, filename))

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
    move_text_files('Used Text Files')

merge_json_files('.', f'{final_json_name}.json')
if not UPLOAD_TO_HF:
    move_completed_json(f'{final_json_name}.json', 'Completed Json Runs')

if UPLOAD_TO_HF:
    import subprocess
    import config
    #You will need to create your own config.py file with your own huggingface ID
    token = config.hf

        # Login command using f-strings for secure variable insertion
    login_command = f"huggingface-cli login --token {token}"

        # Call the subprocess module to execute the login command
    subprocess.run(login_command.split())

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file( #can also do a whole folder upload
        path_or_fileobj= f"{final_json_name}.json",
        path_in_repo= f"{final_json_name}.json",
        repo_id= "YOUR_HF_DATAPATH", #insert the dataset you want to upload in HuggingFace
        repo_type= "dataset",
    )
    move_completed_json(f'{final_json_name}.json', 'Completed Json Runs')