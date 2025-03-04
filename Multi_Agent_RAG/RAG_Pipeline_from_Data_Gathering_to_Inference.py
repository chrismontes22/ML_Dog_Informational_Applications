import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import time
import shutil
import gdown
from typing import List, Dict, Tuple
import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Download the full list of dog breeds (Dog_List.txt)
url = 'https://drive.google.com/file/d/13Va0eqByu_CiR1O3xXO-OZLfJjY8NwwM/view?usp=sharing'
output_path = 'Dog_List.txt'
gdown.download(url, output_path, quiet=False,fuzzy=True)


#############################################################################################################################################
######################################################   Recommended Script Modifiers  ######################################################
#############################################################################################################################################

#List of websites to gather data. You can turn them on or off here by setting to True or False.
DOGTIME: bool = True
DAILYPAWS: bool = True
CANINEJOURNAL: bool = True
PETS4HOMES: bool = True

#Feel free to mess with the chunk size and overlap HERE
chunk_size: int = 500
chunk_overlap: int = 120

"""The Dog_List.txt file is set to work with the standard breed names used in website URL's.
It is highly recomended to refer to the breed name provided in that list."""

#Please insert the dog breeds you would like for gather text data for as a list set to the dog_breed variable.
#If you want to download text data all of the breeds from the Image Classification app, set FULL_LIST to True. Setting to True will override the dog_breed list above it.

dog_breed: List[str] = ["Alaskan-Malamute", "Akita"]
FULL_LIST: bool = False
if FULL_LIST:
    def read_file_to_list(file_path: str) -> List[str]:
        with open(file_path, 'r') as file:
            items_list = [line.strip() for line in file]
        return items_list
    file_path: str = "Dog_List.txt"
    dog_list: List[str] = read_file_to_list(file_path)
    print(dog_list)
if FULL_LIST:
    dog_breed = dog_list
dog_breed = [breed.title() for breed in dog_breed] #This makes the first letter of every word in the list Capitalized to match the dictionary casing (if needed)
directory_path: str = 'Dog_Texts' #Path to save all of the files


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""Most websites have the dog breed standardized. These dictionaries are here (one for each site) for when some dog breeds are not standardized.
In that case, you would have to figure out where the URL error is and fix it in the dictionary in the format below, by comparing the variable in the list to the actual URL.
websites are constantly changing the name used to represent a breed in their URL.
This script logs all the URL errors that occur, so you can let the script run once and then check out the URL log.
You can add your own here if there are other breeds outside of the Dog_List.txt that dont match the website"""

dogtime_W: Dict[str, str] = {"German-Shepherd": "German-Shepherd-Dog", "Xoloitzcuintli": "Xoloitzuintli"}
dailypaws_W: Dict[str, str] = {"Xoloitzcuintli": "Xoloitzcuintli-Mexican-Hairless", "Poodle": "Standard-Poodle"}
caninejournal_W: Dict[str, str] = {"Chinese-Shar-Pei": "Shar-Pei"}
pets4homes_W: Dict[str, str] = {
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


def fetch_and_parse(url: str, dog_breed: str, wnum: int) -> List[str]:
    """Parse the html and output a text file"""
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

def process_and_chunk_text(texts: List[str], file_name: str, x: int = chunk_size, y: int = chunk_overlap) -> None:
    """#This processes the file further, chunks it with Langchain (recursive character)"""
    texts = [line.strip() for line in texts if len(line.strip()) > 55]
    time.sleep(2)
    text_to_write: str = '\n'.join(texts)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=x, chunk_overlap=y)
    chunks: List[str] = text_splitter.split_text(text_to_write)
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
    """Adjustments necessary to text files gathered from Dogtime website."""
    wnum = 1 # This helps name the files that get outputed, such as file1.txt, file1.json
    file_name = f'{dog_breed}{wnum}.txt'
    individual_texts_file_path = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
        dogtime_name = dogtime_W.get(dog_breed, dog_breed).lower() #Retrieves the unique breed name from the site if it isn't like the list. Some sites require all lower cased so this handles with ".lower"
        url = f'https://dogtime.com/dog-breeds/{dogtime_name}' #URL name and variable
        texts = fetch_and_parse(url, dog_breed, wnum)
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



def dailypaws_data(dog_breed: str) -> None:
    """Adjustments necessary to text files gathered from Daily Paws website."""
    wnum: int = 2
    file_name: str = f'{dog_breed}{wnum}.txt'
    individual_texts_file_path: str = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
        dailypaws_name: str = dailypaws_W.get(dog_breed, dog_breed).lower()
        url: str = f'https://www.dailypaws.com/dogs-puppies/dog-breeds/{dailypaws_name}'
        texts: List[str] = fetch_and_parse(url, dog_breed, wnum)
        del texts[1:4]
        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

def caninejournal_data(dog_breed: str) -> None:
    """Adjustments necessary to text files gathered from Canine Journal website."""
    wnum: int = 3
    file_name: str = f'{dog_breed}{wnum}.txt'
    individual_texts_file_path: str = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
        caninejournal_name: str = caninejournal_W.get(dog_breed, dog_breed).lower()
        url: str = f"https://www.caninejournal.com/{caninejournal_name}"
        texts: List[str] = fetch_and_parse(url, dog_breed, wnum)
        texts = texts[5:-5]
        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

def pets4homes_data(dog_breed: str) -> None:
    """Adjustments necessary to text files gathered from Canine Journal website."""
    wnum: int = 4
    file_name: str = f'{dog_breed}{wnum}.txt'
    individual_texts_file_path: str = f"{directory_path}/individual_texts/{dog_breed}{wnum}.txt"
    if not os.path.exists(individual_texts_file_path):
        pets4homes_name: str = pets4homes_W.get(dog_breed, dog_breed).lower()
        url: str = f"https://www.pets4homes.co.uk/dog-breeds/{pets4homes_name}"
        texts: List[str] = fetch_and_parse(url, dog_breed, wnum)
        texts = texts[5:-5]
        process_and_chunk_text(texts, file_name)
    else:
        print(f"File {file_name} already exists.")

##################################################################################
def create_folder_if_not_exists(folder_name: str) -> None:
    """Creates the specified folder if it does not exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_excluded_files() -> List[str]:
    """Returns a list of files to exclude from moving."""
    return ['failed_urls.txt', 'failed_json.txt', 'Dog_List.txt']

def filter_text_files(excluded_files: List[str]) -> List[str]:
    """Filters out excluded files from the list of text files."""
    text_files = [f for f in os.listdir() if f.endswith('.txt')]
    return [f for f in text_files if f not in excluded_files]

def move_file_to_folder(file: str, folder_name: str) -> None:
    """Moves a file to the specified folder, removing it if it already exists."""
    destination_file = os.path.join(folder_name, file)
    if os.path.exists(destination_file):
        os.remove(destination_file)
    shutil.move(file, folder_name)

def move_text_files(folder_name: str) -> None:
    """Moves all text files into a subfolder."""
    create_folder_if_not_exists(folder_name)
    excluded_files = get_excluded_files()
    text_files = filter_text_files(excluded_files)
    for file in text_files:
        move_file_to_folder(file, folder_name)

#Function to merge all files with the same base name into one. It saves the individual files under "individual_texts"
def ensure_directory_exists(directory: str) -> None:
    """Ensures the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def merge_child_files(parent_file: str, child_file: str) -> None:
    """Appends the content of a child file to the parent file."""
    with open(child_file, 'r', encoding='utf-8') as f_child, \
         open(parent_file, 'a', encoding='utf-8') as f_parent:
        f_parent.write(f_child.read() + '\n')

def remove_empty_lines(parent_file: str) -> None:
    """Removes empty lines from the parent file."""
    with open(parent_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(parent_file, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.strip():
                file.write(line)

def move_file_to_used_data(child_file: str, used_data_dir: str) -> None:
    """Moves a child file to the 'used data' directory."""
    shutil.move(child_file, used_data_dir)

def merge_text_files(directory: str, dog_breed: str) -> None:
    """Merges text files by breed."""
    individual_texts_dir = os.path.join(directory, 'individual_texts')
    ensure_directory_exists(individual_texts_dir)

    parent_file = os.path.join(directory, f'{dog_breed}.txt')

    for filename in os.listdir(directory):
        if filename.startswith(dog_breed) and filename != f'{dog_breed}.txt':
            child_file = os.path.join(directory, filename)
            merge_child_files(parent_file, child_file)
            move_file_to_used_data(child_file, individual_texts_dir)

    remove_empty_lines(parent_file)

for breed in dog_breed:
    """Calls the functions for each website, loops for each breed"""
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


smodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="dog")
collection = client.get_or_create_collection("dogdb")

directory_path = f"{directory_path}/individual_texts"
text_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

def read_and_split_file(directory_path: str, file_name: str) -> List[str]:
    """Read a file and split its content into lines."""
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()

def encode_documents(documents: List[str], smodel: SentenceTransformer) -> List[List[float]]:
    """Encode documents into vector embeddings."""
    return smodel.encode(documents)

def extract_breed_and_generate_ids(file_name: str, num_embeddings: int) -> Tuple[str, List[str]]:
    """Extract breed name and generate unique IDs."""
    breed = re.sub(r'\d+\.txt$', '', file_name).replace('_', '-').replace('txt', '')
    base_name = file_name.split('.')[0]
    ids = [f"{base_name}_{i + 1}" for i in range(num_embeddings)]
    return breed, ids

def create_metadatas(breed: str, num_embeddings: int) -> List[dict]:
    """Create metadata entries for each document."""
    return [{"breed": breed} for _ in range(num_embeddings)]

def add_to_collection(collection, documents, ids, vectors, metadatas):
    """Add documents, embeddings, and metadata to the collection."""
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas
    )

def process_text_files(directory_path: str, text_files: List[str], smodel: SentenceTransformer, collection) -> None:
    for file_name in text_files:
        documents = read_and_split_file(directory_path, file_name)
        vectors = encode_documents(documents, smodel)
        num_embeddings = len(vectors)
        breed, ids = extract_breed_and_generate_ids(file_name, num_embeddings)
        metadatas = create_metadatas(breed, num_embeddings)
        add_to_collection(collection, documents, ids, vectors, metadatas)
        print(f"Added {num_embeddings} embeddings from {file_name} to the collection with IDs: {ids}")

process_text_files(directory_path, text_files, smodel, collection)
print("All text files have been processed and added to the collection.")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct").to(device)

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
    """Cleans the text vector output to pass cleanly to the LLM."""
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

