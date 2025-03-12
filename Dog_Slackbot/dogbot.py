import os
import gc
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def initialize_chromadb_and_model():
    """
    Lazily initializes the ChromaDB client, collection, and SentenceTransformer model.
    These are heavy resources that are loaded on demand to save persistent memory.
    """
    client = chromadb.PersistentClient(path="vectordb")
    collection = client.get_or_create_collection('dogdb')
    smodel = SentenceTransformer('./embedmodel')
    return client, collection, smodel

def query_and_retrieve_results(collection, model, question, n_results=3):
    """
    Generates a query, computes embeddings, and retrieves results.
    """
    query = [{'question': question}]
    query_embeddings = model.encode(query)
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )
    return results

def clean_text_block(text):
    """
    Cleans the query results by extracting content between specific keywords.
    """
    start_keyword = "'documents': [["
    end_keyword = "]], 'uris':"

    start_index = text.find(start_keyword)
    end_index = text.find(end_keyword) + len(end_keyword)

    if start_index != -1 and end_index != -1:
        cleaned_text = text[start_index + len(start_keyword):end_index - len(end_keyword)]
        return cleaned_text
    return "Keywords not found in the text."

def configure_generative_ai():
    """
    Configures Google Generative AI using the API key from environment variables.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Google API Key is not set in the environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def generate_human_response(model, results, question, max_tokens=256):
    """
    Generates a response using the generative AI model.
    """
    generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
    prompt = (
        f"""After the colon is a set of text with information about dogs, then a question about the given text. 
Please answer the question based on the text, and do not talk about the documentation:
text - {results}
question - {question}
Respond in a friendly manner; you are an informational assistant about dogs."""
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text

def process_question(question):
    """
    Processes the question and generates a human-readable response.
    Loads heavy resources on demand and cleans them up after processing.
    """
    # Lazy-load heavy resources
    client, collection, smodel = initialize_chromadb_and_model()

    # Query the collection and process the results
    results = query_and_retrieve_results(collection, smodel, question)
    cleaned_results = clean_text_block(str(results))
    
    # Configure and load the generative AI model on demand
    genai_model = configure_generative_ai()
    human_response = generate_human_response(genai_model, cleaned_results, question)
    
    # Cleanup heavy objects to free up RAM
    del smodel, client, collection, genai_model
    gc.collect()
    
    return human_response

# Initialize the Slack Bolt app (lightweight)
app = App(token=SLACK_BOT_TOKEN)
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for Slack mentions.
    When the bot is mentioned, this function extracts the question, 
    processes it by loading heavy models on demand, and sends the response.
    """
    # Extract the message text from the Slack event
    text2 = body["event"]["text"]

    # Remove the bot mention from the message
    mention = f"<@{SLACK_BOT_USER_ID}>"
    question = text2.replace(mention, "").strip()

    # Respond immediately to the mention
    say("*Woof!*")
    
    # Process the question using the lazy-loaded resources
    human_response = process_question(question)
    
    # Send the processed response back to the channel
    say(human_response)
    say("*Woof!*")

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return data.get("challenge"), 200, {'Content-Type': 'text/plain'}
    # Process other event types via SlackRequestHandler
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5000)
