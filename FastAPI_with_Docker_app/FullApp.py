import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torchvision.models as models
import torchvision.transforms as v2
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and modify it
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Serve static files from the current directory
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
async def main() -> None:
    """Import the content of the index.html file"""
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

def save_uploaded_file(file: UploadFile) -> Path:
    """Save the uploaded file temporarily to disk."""
    image_path = Path(f"temp_{file.filename}")
    with open(image_path, "wb") as f:
        f.write(file.file.read())  # Read and write file in binary mode
    return image_path

def transform_image(image_path: Path) -> torch.Tensor:
    """Transform and preprocess the image for the model."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    transformed_img = transforms_test(image).unsqueeze(0).to(device)
    return transformed_img

def perform_inference(transformed_img: torch.Tensor) -> tuple:
    """Perform inference using the model and return predictions."""
    output = model(transformed_img)
    output_softmax = torch.nn.functional.softmax(output, dim=1)
    topk_values, topk_indices = torch.topk(output_softmax, 3)
    return topk_values, topk_indices

def extract_predictions(topk_values: torch.Tensor, topk_indices: torch.Tensor) -> tuple:
    """Extract top-k prediction labels and probabilities."""
    topk_indices = topk_indices.tolist()[0]
    topk_labels = [labels[index] for index in topk_indices]
    topk_values = topk_values.tolist()[0]
    return topk_labels, topk_values

def generate_dynamic_content(topk_labels: list, topk_values: list) -> dict:
    """Generate dynamic content like the predictions list and dropdown options."""
    predictions_list = "".join(
        f"<li>{label.replace('-', ' ')}{breed_nicknames.get(label, '')}: {prob:.2%}</li>"
        for label, prob in zip(topk_labels, topk_values)
    )
    dropdown_options = "".join(
        f'<option value="{i+1}">{label.replace("-", " ")}{breed_nicknames.get(label, "")}</option>'
        for i, label in enumerate(topk_labels)
    )
    return {
        "predictions_list": predictions_list,
        "dropdown_options": dropdown_options,
        "topk_labels": ",".join(topk_labels),
        "topk_values": ",".join(map(str, topk_values)),
    }

def generate_predictions_html(topk_labels: list, topk_values: list) -> str:
    """Generate the full HTML content by combining the template with dynamic content."""
    dynamic_content = generate_dynamic_content(topk_labels, topk_values)
    with open("index2.html", "r", encoding="utf-8") as f:
        predictions_html = f.read()
        
    for key, value in dynamic_content.items():
        predictions_html = predictions_html.replace(f"{{{{{key}}}}}", value)
    
    return predictions_html

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict dog breeds from an uploaded image."""
    # Save the uploaded file temporarily
    image_path = save_uploaded_file(file)
    try:
        transformed_img = transform_image(image_path)
        topk_values, topk_indices = perform_inference(transformed_img)
        topk_labels, topk_values = extract_predictions(topk_values, topk_indices)
        predictions_html = generate_predictions_html(topk_labels, topk_values)
        return JSONResponse(content={"predictions_html": predictions_html})
    finally:
        # Clean up temporary file
        os.remove(image_path)


def generate_custom_breed_html() -> str:
    """Generate HTML for entering a custom breed."""
    return """
    <h1>Please enter your dog's breed:</h1>
    <form onsubmit="submitCustomBreed(event)">
        <div class="form-group">
            <input type="text" name="custom_breed" maxlength="50" required class="form-control">
        </div>
        <button type="submit" class="btn btn-warning">Enter Breed</button>
    </form>
    """

def generate_known_breed_html(dog_breed: str) -> str:
    """Generate HTML for a known dog breed."""
    dog_breed_cleaned = dog_breed.replace('-', ' ')
    return f"""
    <h1>What a cute {dog_breed_cleaned}!</h1>
    <form onsubmit="askQuestion(event)">
        <div class="form-group">
            <label for="question">What would you like to know about the {dog_breed_cleaned} breed?</label>
            <input type="text" name="question" id="question" required class="form-control">
        </div>
        <input type="hidden" name="dog_breed" value="{dog_breed}">
        <button type="submit" class="btn btn-secondary">Ask</button>
    </form>
    <div id="question_response"></div>
    """

@app.post("/feedback/")
async def feedback(choice: int = Form(...), topk_labels: str = Form(...)):
    """Handle feedback depending on the user's choice."""
    topk_labels_list = topk_labels.split(",")
    
    if choice == 4:
        # Generate HTML for entering a custom breed
        feedback_html = generate_custom_breed_html()
    else:
        # Generate HTML for a known dog breed
        dog_breed = topk_labels_list[choice - 1]
        feedback_html = generate_known_breed_html(dog_breed)
    
    # Return the feedback HTML as a JSON response
    return JSONResponse(content={"feedback_html": feedback_html})

# Endpoint to handle custom breed input
def generate_feedback_html(dog_breed: str) -> str:
    """Generate the HTML content for the feedback based on the dog breed."""
    dog_breed_cleaned = dog_breed.strip().replace('-', ' ')
    return f"""
    <h1>What a cute {dog_breed_cleaned}!</h1>
    <form onsubmit="askQuestion(event)">
        <div class="form-group">
            <label for="question">What would you like to know about the {dog_breed_cleaned} breed?</label>
            <input type="text" name="question" id="question" required class="form-control">
        </div>
        <input type="hidden" name="dog_breed" value="{dog_breed.strip()}">
        <button type="submit" class="btn btn-secondary">Ask</button>
    </form>
    <div id="question_response"></div>
    """

@app.post("/custom_breed/")
async def custom_breed(custom_breed: str = Form(...)):
    """Endpoint to process custom dog breed and generate feedback HTML."""
    dog_breed = custom_breed.strip()
    feedback_html = generate_feedback_html(dog_breed)
    return JSONResponse(content={"feedback_html": feedback_html})

def initialize_models():
    """Load or save models and tokenizers as needed."""
    if not os.path.exists("./hfmodel"): 
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
        tokenizer.save_pretrained("./hfmodel")
        model_llm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct").to(device)
        model_llm.save_pretrained("./hfmodel")
    else:
        tokenizer = AutoTokenizer.from_pretrained("./hfmodel")
        model_llm = AutoModelForCausalLM.from_pretrained("./hfmodel").to(device)
    
    sentence_model = SentenceTransformer('./embed')
    return tokenizer, model_llm, sentence_model

def query_chromadb(sentence_model, dog_breed, question, n_results=4):
    """ Initialize cHromaDB, embed query, and retrieve results from vector store"""
    client = chromadb.PersistentClient(path="vectordb") # Initialize the chromadb client and collection
    collection = client.get_or_create_collection("dogdb")

    query = [{'question': f"{dog_breed}: {question}?"}] # Create query embeddings
    query_embeddings = sentence_model.encode(query)  

    results = collection.query(     # Query the database and clean the results
        query_embeddings=query_embeddings,
        n_results=n_results
    )
    return clean_text_block(str(results))


def clean_text_block(text):
    """Extract and clean the relevant text block from the results."""
    start_keyword = "'documents': [["
    end_keyword = "]], 'uris':"

    start_index = text.find(start_keyword)
    end_index = text.find(end_keyword) + len(end_keyword)

    if start_index != -1 and end_index != -1:
        cleaned_text = text[start_index + len(start_keyword):end_index - len(end_keyword)]
        return cleaned_text
    else:
        return "Keywords not found in the text."

def prepare_llm_input(tokenizer, text, question):
    """Prepare the input for the language model."""
    messages = [{
        "role": "user",
        "content": f"""After the colon is a set of text with information about dogs, then a question about the given text. Please answer the question based off the text, and do not talk about the documentation:
        text - {text}
        question - {question}
        Respond in a friendly manner; you are an informational about dogs."""
    }]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    return inputs

def generate_response(model_llm, inputs, tokenizer):
    """Generate a response using the LLM."""
    outputs = model_llm.generate(inputs, max_new_tokens=150, temperature=0.4, top_p=0.6, do_sample=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_assistant_response(output_text)

def extract_assistant_response(text):
    """Extract only the response text after 'assistant'."""
    keyword = "assistant"
    start_idx = text.find(keyword)
    
    if start_idx != -1:
        cleaned_text = text[start_idx + len(keyword):].strip()
        cleaned_text = cleaned_text.strip('"').strip()
        return cleaned_text
    else:
        return "No response found."

async def process_question(dog_breed: str, question: str) -> str:
    """Process the question by querying the database and generating a response."""
    # Initialize models and query the chromadb for relevant data
    tokenizer, model_llm, sentence_model = initialize_models()
    results = query_chromadb(sentence_model, dog_breed, question)

    # Prepare LLM input and generate response
    inputs = prepare_llm_input(tokenizer, results, question)
    cleaned_output = generate_response(model_llm, inputs, tokenizer)

    return cleaned_output

@app.post("/ask/")
async def ask_question(dog_breed: str = Form(...), question: str = Form(...)):
    """Main endpoint to handle the ask question functionality."""
    cleaned_output = await process_question(dog_breed, question)

    # Create the response HTML
    response_html = f"""
    <p>Loading Complete. Response:</p>
    <p>{cleaned_output}</p>
    <p>Feel free to ask another question, change the picture, or select another breed!</p>
    """

    # Return the response as JSON
    return JSONResponse(content={"response_html": response_html})


# Command: `uvicorn script_name:app --reload`
