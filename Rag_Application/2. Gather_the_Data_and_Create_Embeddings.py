import re
import os
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the model
smodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the chromadb client and collection
client = chromadb.PersistentClient(path="dog")
collection = client.get_or_create_collection("dogdb")

directory_path = "Dogs/used_data" #I Recommend using the individual files in the used_data folder instead of the merged files of simply Dogs
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

