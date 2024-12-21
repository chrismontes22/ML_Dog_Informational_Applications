import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists("./hf_model"): 
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
    tokenizer.save_pretrained("./hf_model")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct").to(device)
    model.save_pretrained("./hf_model")
else:
    tokenizer = AutoTokenizer.from_pretrained("./hf_model")
    model = AutoModelForCausalLM.from_pretrained("./hf_model").to(device)

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
    n_results=3 # how many results to return
)
print(results)   
# Run this line above if you would like to see the retrieved data

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

