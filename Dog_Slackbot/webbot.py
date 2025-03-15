import os
import threading
from typing import Any, Dict, Tuple
from dotenv import find_dotenv, load_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from tavily import TavilyClient
import google.generativeai as genai
from flask import jsonify
import requests


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN: str = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET: str = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_BOT_USER_ID: str = os.environ.get("SLACK_BOT_USER_ID")

if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET or not SLACK_BOT_USER_ID:
    raise EnvironmentError("Slack credentials are not set in the environment variables.")


def configure_generative_ai() -> genai.GenerativeModel:
    """
    Configures Google Generative AI using the API key from environment variables.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Google API Key is not set in the environment variables")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def generate_human_response(model: genai.GenerativeModel, question: str, max_tokens: int = 256) -> str:
    """
    Generates a response using the generative AI model.
    """
    generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
    prompt = (
        f"""You are a friendly assistant.
        Please answer the following question based on the text or to your best knowledge, but do not mention the text:
        question - {question}"""
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text

def init_tavily_client() -> TavilyClient:
    """
    Initializes the TavilyClient using the API key from environment variables.
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise EnvironmentError("Tavily API Key is not set in the environment variables")

    return TavilyClient(api_key=tavily_api_key)

def call_tavily_api(query: str) -> Dict[str, Any]:
    """
    Calls the Tavily API using TavilyClient to retrieve results based on the user's query.
    """
    tavily_client = init_tavily_client()
    response = tavily_client.search(query)

    if not response or "results" not in response:
        raise Exception("Failed to retrieve results from Tavily API")

    return response

def summarize_tavily_results(results: list) -> str:
    """
    Summarizes the content from the Tavily results if no 'answer' is provided.
    """
    if not results:
        return "Sorry, I couldn't find any relevant information."

    # Extract content from the top results and summarize it
    summary = []
    for result in results[:2]:  # Limit to the top 2 results for brevity
        title = result.get("title", "Untitled")
        content = result.get("content", "")
        url = result.get("url", "")
        summary.append(f"*{title}*\n{content}\nRead more: {url}\n")

    return "\n".join(summary)

def process_question_with_tavily_and_google(query: str) -> str:
    """
    Processes the user's query by calling Tavily, summarizing the results if needed,
    and passing them to Google Generative AI for refinement.
    """
    # Step 1: Call Tavily API
    response = call_tavily_api(query)

    # Step 2: Extract or summarize results
    if response.get("answer"):
        cleaned_results = response["answer"]
    else:
        cleaned_results = summarize_tavily_results(response.get("results", []))

    # Step 3: Use cleaned results to query Google Generative AI
    genai_model = configure_generative_ai()
    human_response = generate_human_response(genai_model, cleaned_results)
    
    return human_response


# Initialize Slack app and Flask app
app = App(token=SLACK_BOT_TOKEN)
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@app.command("/webbot")
def handle_websearch(ack, body: Dict[str, Any], say) -> None:
    """Takes user input and cleans it, passes input through tavily and google."""
    # Immediately ack with a quick response
    ack("*Fetching your data...*")
    query: str = body.get("text", "").strip()
    response_url = body.get("response_url")
    
    # Process in a separate thread
    def background_process():
        try:
            final_response: str = process_question_with_tavily_and_google(query)
            payload = {"text": final_response}
            requests.post(response_url, json=payload)
        except Exception as e:
            payload = {"text": f"An error occurred: {str(e)}"}
            requests.post(response_url, json=payload)
    
    threading.Thread(target=background_process).start()


@flask_app.route("/webbot", methods=["POST"])
def handle_websearch_command():
    """
    Handles the Slack slash command "/webbot" via Flask.
    """
    data = request.form  # Slash commands send data as form-encoded
    query = data.get("text", "").strip()
    response_url = data.get("response_url")  # URL to send delayed response

    # Immediately acknowledge the command
    ack_response = {"text": "*Fetching your data...*", "response_type": "ephemeral"}
    requests.post(response_url, json=ack_response)  

    # Process query in the background
    def background_process():
        try:
            final_response = process_question_with_tavily_and_google(query)
            requests.post(response_url, json={"text": final_response})
        except Exception as e:
            requests.post(response_url, json={"text": f"An error occurred: {str(e)}"})

    threading.Thread(target=background_process).start()

    return jsonify({"response_type": "ephemeral", "text": "Processing your request..."}), 200


@app.event("app_mention")
def handle_mentions(body: Dict[str, Any], say: Any) -> None:
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    # Extract the message text from the Slack event
    text2 = body["event"]["text"]

    # Remove the bot mention from the message to get the query
    mention = f"<@{SLACK_BOT_USER_ID}>"  # Your bot's user ID
    query = text2.replace(mention, "").strip()

    # Respond immediately to acknowledge the mention
    say("*Woof* Fetching your data...")

    # Process the query in a separate thread to avoid timeout
    def background_process():
        try:
            # Use the same processing pipeline as the slash command
            final_response: str = process_question_with_tavily_and_google(query)
            say(final_response)  # Send the response back to Slack
        except Exception as e:
            say(f"An error occurred: {str(e)}")

    threading.Thread(target=background_process).start()


@flask_app.route("/slack/events", methods=["POST"])
def slack_events() -> Tuple[str, int, Dict[str, str]]:
    """Handle Slack events via Flask"""
    data = request.json
    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return data.get("challenge"), 200, {'Content-Type': 'text/plain'}
    # Process other event types via SlackRequestHandler
    return handler.handle(request)

# Run the Flask app
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5000)
