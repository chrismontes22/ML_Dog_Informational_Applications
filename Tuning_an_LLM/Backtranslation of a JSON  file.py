"""This code is for translating and back-translating for data augmentation. While I have integrated it into the main text pipeline, I also have it separate because when I run a large number of labels,
this code takes too long. It is device agnostic, so a GPU is highly recommended."""
#Because of the way my json data is structured, you need to save all the breeds which you want to data augment into a single JSON file, and you will also need to specify the breeds in that JSON file.

import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Load the list of dog breeds
dog_breeds = ["Alaskan-Malamute","Staffordshire-Bull-Terrier",'Cane-Corso','Irish-Wolfhound','Akita','Norwegian-Elkhound','Chow-Chow', 'Siberian-Husky','Lhasa-Apso', 'French-Bulldog','American-Foxhound' ]  # your list of dog breeds

# Load the JSON file
with open('Data_to translate.json') as f:
    data = json.load(f)

# Start timing
start_time = time.time()

# Translate and back-translate the text for each breed
for dog_breed in dog_breeds:
    for item in data:
        if item['text'].startswith(dog_breed + ':'):
            text_to_translate = item['text'][len(dog_breed) + 1:]
            input_ids = tokenizer.encode(text_to_translate, return_tensors='pt').to(device)  # Move to GPU
            outputs = model.generate(input_ids=input_ids, max_length=250, min_length=50, no_repeat_ngram_size=3, num_beams=5, early_stopping= True)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            input_ids = tokenizer.encode(translated_text, return_tensors='pt').to(device)  # Move to GPU
            outputs = model.generate(input_ids=input_ids, max_length=250, min_length=50, no_repeat_ngram_size=3, num_beams=5, early_stopping= True)
            back_translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            item['text'] = dog_breed + ': ' + back_translated_text

# Write the translated data to a new JSON file
with open('translated_data.json', 'w') as f:
    json.dump(data, f, indent=4)

# End timing
end_time = time.time()

# Print the elapsed time
print(f"Elapsed time: {end_time - start_time} seconds")