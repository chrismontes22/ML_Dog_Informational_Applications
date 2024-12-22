"""
Original file is located at
    https://colab.research.google.com/drive/1by5UTMttZwW6xGGo89hmVNu2b90V3-HV

Here is a full pipeline to create a basic RAG application. By running this script, you can download information about dogs, parse it, and organize/chunk it. Then it embeds the information into a vector store. Finally, it downloads an LLM, retrieves the appropriate embeddings, and outputs a fact, all in a single click.
"""

#This notebook was created in Google Colab, but works well as a single script. The main difference is that here it is recomended to install the necessary packages before running the script. Here are the pip packages:
#pip install requests langchain bs4 gdown transformers sentence-transformers chromadb

"""The code below downloads all the text data in HTML format and cleans it up for embedding. Here you can modify the list of dog breeds and mess around with the chunk size.
You can always add more breeds after you create the vectore store! To do so go to the dog_breed list and simply insert the breeds you want the data from, and run everything again. You can always move the vector around as a folder, just make sure to keep track of the vector database name. It is "dogdb" here, which can be found in the line:
collection = client.get_or_create_collection("dogdb")
You use this line to both create and call the vector database, so you need to track this name.

The script is a modified version of my [Text_Data_Pipeline.py](https://github.com/chrismontes22/Dog-Classification/blob/main/Tuning_an_LLM/Text%20Data%20Pipeline.py) that was used to streamline the fine tuning of an LLM.
Instead of using Bert for data augmentation and turning the file into JSON,
it chunks the data with Langchain and organizes it by dog breed after having downloaded and parsed the text. Most of the customization you will need in this script is int the section
labeled "Recommended Script Modifiers".
"""

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import os
import time
import shutil
import gdown

# Download the full list of dog breeds (Dog_List.txt)
url = 'https://drive.google.com/file/d/13Va0eqByu_CiR1O3xXO-OZLfJjY8NwwM/view?usp=sharing'
output_path = 'Dog_List.txt'
gdown.download(url, output_path, quiet=False,fuzzy=True)

#############################################################################################################################################
######################################################   Recommended Script Modifiers  ######################################################
#############################################################################################################################################

#List of websites to gather data. You can turn them on or off here by setting to True or False.
DOGTIME = True
DAILYPAWS = True
CANINEJOURNAL = False
PETS4HOMES = True

#Feel free to mess with the chunk size and overlap HERE
chunk_size = 500
chunk_overlap = 120

"""The Dog_List.txt file is set to work with the standard breed names used in website URL's.
It is highly recomended to refer to the breed name provided in that list."""

#Please insert the dog breeds you would like for gather text data for as a list set to the dog_breed variable.
#If you want to download text data all of the breeds from the Image Classification app, set FULL_LIST to True. Setting to True will override the dog_breed list above it.

dog_breed = ["Siberian-Husky", "Labrador-Retriever"]
FULL_LIST = False
if FULL_LIST:
    def read_file_to_list(file_path):
        with open(file_path, 'r') as file:
            items_list = [line.strip() for line in file]
        return items_list
    file_path = "Dog_List.txt"
    dog_list = read_file_to_list(file_path)
    print(dog_list)
if FULL_LIST:
    dog_breed = dog_list
dog_breed = [breed.title() for breed in dog_breed] #This makes the first letter of every word in the list Capitalized to match the dictionary casing (if needed)
directory_path= 'Dog_Texts' #Path to save all of the files


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""Most websites have the dog breed standardized. These dictionaries are here (one for each site) for when some dog breeds are not standardized.
In that case, you would have to figure out where the URL error is and fix it in the dictionary in the format below, by comparing the variable in the list to the actual URL.
websites are constantly changing the name used to represent a breed in their URL.
This script logs all the URL errors that occur, so you can let the script run once and then check out the URL log.
You can add your own here if there are other breeds outside of the Dog_List.txt that dont match the website"""

dogtime_W = {"German-Shepherd": "German-Shepherd-Dog", "Xoloitzcuintli": "Xoloitzuintli"}
dailypaws_W = {"Xoloitzcuintli": "Xoloitzcuintli-Mexican-Hairless", "Poodle": "Standard-Poodle"}
caninejournal_W = {"Chinese-Shar-Pei": "Shar-Pei"}
pets4homes_W = {
    "Chinese-Shar-Pei": "Shar-Pei",
    "Shiba-Inu": "Japanese-Shiba-Inu",
    "Belgian-Malinois": "Belgian-Shepherd-Dog",
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
    time.sleep(2) #This slows down the script so that it doesn't create too many calls to the website at once.

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
Make sure to add the line for wnum, this is to identify the text file to the corresponding website; example the next website would have wnum = 5"""

def dogtime_data(dog_breed):
    wnum = 1 #This helps name the files that get outputed, such as file1.txt, file1.json
    file_name = f'{dog_breed}{wnum}.txt'
    individual_texts_file_path = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
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
    individual_texts_file_path = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
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
    individual_texts_file_path = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
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
    individual_texts_file_path = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
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
    excluded_files = ['failed_urls.txt', 'failed_json.txt', 'Dog_List.txt']
    text_files = [f for f in text_files if f not in excluded_files]

    # Move each text file into the folder
    for file in text_files:
        destination_file = os.path.join(folder_name, file)
        if os.path.exists(destination_file):
            os.remove(destination_file)  # Remove the file if it already exists
        shutil.move(file, folder_name)

#Function to merge all files with the same base name into one. It saves the individual files under "individual_texts"
def merge_text_files(directory, dog_breed):
    # Ensure the 'used data' folder exists
    individual_texts_dir = os.path.join(directory, 'individual_texts')
    if not os.path.exists(individual_texts_dir):
        os.makedirs(individual_texts_dir)

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
            shutil.move(child_file, individual_texts_dir)

    with open(parent_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(parent_file, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.strip():
                file.write(line)

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
    move_text_files(directory_path)
    merge_text_files(directory_path, breed)

"""Here we create the vector store and embeddings from the text files from the code above. The code above created a text file from each website for each breed, so potentially 4 files per breed. They are saved in the "Dogs/individual_texts" folder. I recommend embedding these text files, instead of the merged text files by breed that are saved simply under "Dog_Texts"."""

import re
import os
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the model
smodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the chromadb client and collection
client = chromadb.PersistentClient(path="dog")
collection = client.get_or_create_collection("dogdb")

directory_path = f"{directory_path}/individual_texts"
text_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

for file_name in text_files:
    # Construct the full path to the text file
    file_path = os.path.join(directory_path, file_name)

    # Open and read the corresponding text file
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()


    # Assuming each line in the file represents an embedding/document
    documents = file_content.splitlines()

    # Encode the documents
    vectors = smodel.encode(documents)

    # Generate IDs for each document
    num_embeddings = len(vectors)

    # Extract breed name using regex to remove numbers and file extensions
    breed = re.sub(r'\d+\.txt$', '', file_name)  # Removes the trailing number and .txt
    breed = breed.replace('_', '-').replace('txt', '')  # Convert underscores to hyphens if needed

    ids = [f"{file_name.split('.')[0]}_{i + 1}" for i in range(num_embeddings)]  # Prefix IDs with file name

    # Add metadata for each document
    metadatas = [{"breed": breed} for _ in range(num_embeddings)]  # Example metadata

    # Add the documents, embeddings, and metadata to the collection
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas
    )

    # Print confirmation
    print(f"Added {num_embeddings} embeddings from {file_name} to the collection with IDs: {ids}")

print("All text files have been processed and added to the collection.")

"""The following code downloads the LLM for inference later on. Feel free to change the model. To do so, find an appropriate model from Hugging Face, and replace HuggingFaceTB/SmolLM2-1.7B-Instruct with repo/model-name. Make sure it is a text-generation model and you have the appropriate permissions set."""

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct").to(device)

"""The final block allows the user to input a query, then retrieves the appropriate data from the vector store. Finally the chosen LLM cleans it for inference."""

# Initialize the model
smodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# Initialize the chromadb client and collection
client = chromadb.PersistentClient(path="dog")
collection = client.get_or_create_collection("dogdb")

question = input("Please enter your question: ")

query = [
    {'question': f"{question}"},
]
query_embeddings= smodel.encode(query, device = device)
results = collection.query(
    query_embeddings=query_embeddings,
    n_results=3 # how many results to return. You can change this, but three works well
)
print(results)
#Prints out the embeddign results. If you only want inference, cancel this line

def clean_text_block(text):
    start_keyword = "'documents': [["
    end_keyword = "]], 'uris':"

    start_index = text.find(start_keyword)
    end_index = text.find(end_keyword) + len(end_keyword)

    if start_index != -1 and end_index != -1:
        cleaned_text = text[start_index + len(start_keyword):end_index - len(end_keyword)]
        return cleaned_text
    else:
        return "Keywords not found in the text."

results = clean_text_block(str(results))


messages = [{"role": "user", "content": f"""After the colon is a set of text with information about dogs, then a question about the given text. Please answer the question based off the text, and do not talk about the documentation:
text - {results}
question - {question}
Respond in a friendly manner; you are an informational about dogs."""}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=150, temperature=0.4, top_p=0.6, do_sample=True)
output_text = tokenizer.decode(outputs[0])
start_index = output_text.find("<|im_start|>assistant") + len("<|im_start|>assistant")
end_index = output_text.find("<|im_end|>", start_index)
print(output_text[start_index:end_index].strip())