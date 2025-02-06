import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as v2
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and modify it
device = torch.device("cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 73)
model.load_state_dict(torch.load('Dogrun2.pth', map_location=device))
model.eval()
model.to(device)

# Load breed labels and nicknames
with open('Dog_List.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

breed_nicknames = {
    'Xoloitzcuintli': ' (Mexican Hairless)',
    'Staffordshire-Bull-Terrier': ' (Pitbull)',
    'Pembroke-Welsh-Corgi': ' (Corgi)',
}

# Transformations for the image
transforms_test = v2.Compose([
    v2.Resize((224, 224)),
    v2.CenterCrop((224, 224)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# HTML form for uploading an image
@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head>
            <title>Dog Breed Classifier</title>
            <script>
                async function uploadImage(event) {
                    event.preventDefault();
                    const formData = new FormData();
                    formData.append('file', document.querySelector('input[type="file"]').files[0]);

                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    document.getElementById('predictions').innerHTML = result.predictions_html;
                }

                async function submitFeedback(event) {
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    const response = await fetch('/feedback/', {
                        method: 'POST',
                        body: new URLSearchParams(formData)
                    });

                    const result = await response.json();
                    document.getElementById('feedback').innerHTML = result.feedback_html;
                }

                async function askQuestion(event) {
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    const response = await fetch('/ask/', {
                        method: 'POST',
                        body: new URLSearchParams(formData)
                    });

                    const result = await response.json();
                    document.getElementById('question_response').innerHTML = result.response_html;
                }

                async function submitCustomBreed(event) {
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    const response = await fetch('/custom_breed/', {
                        method: 'POST',
                        body: new URLSearchParams(formData)
                    });

                    const result = await response.json();
                    document.getElementById('feedback').innerHTML = result.feedback_html;
                }
            </script>
        </head>
        <body>
            <h1>Upload an image of your dog</h1>
            <form onsubmit="uploadImage(event)">
                <input type="file" name="file">
                <button type="submit">Upload</button>
            </form>
            <div id="predictions"></div>
            <div id="feedback"></div>
            <div id="question_response"></div>
        </body>
    </html>
    """

# Endpoint to handle predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    image_path = Path(f"temp_{file.filename}")
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Open and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_img = transforms_test(image).unsqueeze(0).to(device)

    # Perform inference
    output = model(transformed_img)
    output_softmax = torch.nn.functional.softmax(output, dim=1)
    topk_values, topk_indices = torch.topk(output_softmax, 3)

    # Get top-3 predictions and probabilities
    topk_indices = topk_indices.tolist()[0]
    topk_labels = [labels[index] for index in topk_indices]
    topk_values = topk_values.tolist()[0]

    # Clean up temporary file
    os.remove(image_path)

    # Generate response HTML with predictions and dropdown menu for feedback
    predictions_html = "<h1>Top-3 Predicted Dog Breeds:</h1><ul>"
    for i, (label, prob) in enumerate(zip(topk_labels, topk_values)):
        nickname = breed_nicknames.get(label, "")
        predictions_html += f"<li>{label.replace('-', ' ')}{nickname}: {prob:.2%}</li>"
    
    predictions_html += """
        </ul>
        <form onsubmit="submitFeedback(event)">
            <label for="choice">Choose the correct breed:</label><br>
            <select name="choice" id="choice">
    """
    
    # Add the predicted labels as options in the dropdown menu
    for i, label in enumerate(topk_labels):
        nickname = breed_nicknames.get(label, "")
        predictions_html += f'<option value="{i+1}">{label.replace("-", " ")}{nickname}</option>'
    
    predictions_html += """
                <option value="4">None of these</option>
            </select><br><br>
            <div id="custom_breed_div" style="display: none;">
                If none of these: Enter your dog's breed: 
                <input type="text" name="custom_breed" maxlength="50" id="custom_breed_input"><br><br>
            </div>
    """
    
    # Pass the top-k labels and values as hidden inputs
    predictions_html += f"""
            <input type="hidden" name="topk_labels" value="{','.join(topk_labels)}">
            <input type="hidden" name="topk_values" value="{','.join(map(str, topk_values))}">
            <button type="submit">Enter</button>
        </form>
        <script>
            const choiceSelect = document.querySelector('#choice');
            const customBreedDiv = document.querySelector('#custom_breed_div');
            choiceSelect.addEventListener('change', function() {{
                if (choiceSelect.value === '4') {{
                    customBreedDiv.style.display = 'block';
                }} else {{
                    customBreedDiv.style.display = 'none';
                }}
            }});
        </script>
    """
    
    return JSONResponse(content={"predictions_html": predictions_html})

# Endpoint to handle user feedback
@app.post("/feedback/")
async def feedback(
    choice: int = Form(...),
    topk_labels: str = Form(...)
):
    topk_labels_list = topk_labels.split(",")
    
    if choice == 4:
        feedback_html = """
        <h1>Please enter your dog's breed:</h1>
        <form onsubmit="submitCustomBreed(event)">
            <input type="text" name="custom_breed" maxlength="50" required>
            <button type="submit">Enter Breed</button>
        </form>
        """
        return JSONResponse(content={"feedback_html": feedback_html})
    else:
        dog_breed = topk_labels_list[choice - 1]
        feedback_html = f"""
        <h1>What a cute {dog_breed.replace('-', ' ')}!</h1>
        <form onsubmit="askQuestion(event)">
            <label for="question">What would you like to know about {dog_breed.replace('-', ' ')}s?</label><br>
            <input type="text" name="question" id="question" required><br><br>
            <input type="hidden" name="dog_breed" value="{dog_breed}">
            <button type="submit">Ask</button>
        </form>
        <div id="question_response"></div>
        """
        return JSONResponse(content={"feedback_html": feedback_html})

# Endpoint to handle custom breed input
@app.post("/custom_breed/")
async def custom_breed(custom_breed: str = Form(...)):
    dog_breed = custom_breed.strip()
    feedback_html = f"""
    <h1>What a cute {dog_breed.replace('-', ' ')}!</h1>
    <form onsubmit="askQuestion(event)">
        <label for="question">What would you like to know about {dog_breed.replace('-', ' ')}s?</label><br>
        <input type="text" name="question" id="question" required><br><br>
        <input type="hidden" name="dog_breed" value="{dog_breed}">
        <button type="submit">Ask</button>
    </form>
    <div id="question_response"></div>
    """
    return JSONResponse(content={"feedback_html": feedback_html})


# Endpoint to handle user's question about the dog breed
@app.post("/ask/")
async def ask_question(
    dog_breed: str = Form(...),
    question: str = Form(...)
):
    # Load the tokenizer and model
    if not os.path.exists("./hf_model"): 
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
        tokenizer.save_pretrained("./hf_model")
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct").to(device)
        model.save_pretrained("./hf_model")
    else:
        tokenizer = AutoTokenizer.from_pretrained("./hf_model")
        model = AutoModelForCausalLM.from_pretrained("./hf_model").to(device)

    smodel = SentenceTransformer('./embed')

    # Initialize the chromadb client and collection
    client = chromadb.PersistentClient(path="vectordb")
    collection = client.get_or_create_collection("dogdb")

    # Create the query and get embeddings
    query = [{'question': f"{dog_breed}: {question}?"}]
    query_embeddings = smodel.encode(query)
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=4  # Number of results to return
    )

    # Clean the text block
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

    # Prepare the input for the LLM
    messages = [{
        "role": "user",
        "content": f"""After the colon is a set of text with information about dogs, then a question about the given text. Please answer the question based off the text, and do not talk about the documentation:
        text - {results}
        question - {question}
        Respond in a friendly manner; you are an informational about dogs."""
    }]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=150, temperature=0.4, top_p=0.6, do_sample=True)

    # Decode the output while skipping special tokens
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Function to extract only the text after "assistant"
    def extract_assistant_response(text):
        keyword = "assistant"
        start_idx = text.find(keyword)
        
        if start_idx != -1:
            # Extract everything after "assistant"
            cleaned_text = text[start_idx + len(keyword):].strip()
            # Remove any leading/trailing whitespace or unwanted characters
            cleaned_text = cleaned_text.strip('"').strip()
            return cleaned_text
        else:
            return "No response found."

    # Extract the assistant's response
    cleaned_output = extract_assistant_response(output_text)

    # Return only the prompt and the cleaned output
    response_html = f"""
    <h1>What would you like to know about {dog_breed}?</h1>
    <p>{cleaned_output}</p>
    """
    return JSONResponse(content={"response_html": response_html})

# Run the FastAPI app using Uvicorn
# Command: `uvicorn script_name:app --reload`