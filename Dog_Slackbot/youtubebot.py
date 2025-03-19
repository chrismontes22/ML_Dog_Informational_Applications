import os
import threading
from typing import Any, Dict, Tuple
from dotenv import find_dotenv, load_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
import requests

# Scopes: app_mentions:read, chat:write, commands
# Events: app_mention


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN: str = os.environ.get("SLACK_BOT_TOKEN")
SLACK_BOT_USER_ID: str = os.environ.get("SLACK_BOT_USER_ID")
YOUTUBE_API_KEY: str = os.environ.get("YOUTUBE_API_KEY")

# Create the YouTube API client
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Initialize Slack app and Flask app
app = App(token=SLACK_BOT_TOKEN)
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

def search_youtube(query, max_results=3):
    """
    Searches YouTube for videos matching a specific query.

    Args:
        query (str): The search query term.
        max_results (int): The maximum number of results to retrieve.

    Returns:
        list: A list of video items retrieved from the search query, or None if an error occurred.
    """
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",  # Ensures only video results are retrieved
            maxResults=max_results,
        )
        response = request.execute()
        return response["items"]
    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return None


def display_video_details(video_items):
    """
    Displays the details of a list of YouTube videos.

    Args:
        video_items (list): A list of video items retrieved from the search query.

    Returns:
        str: A formatted string with video details.
    """
    try:
        result = []
        for item in video_items:
            # Safely retrieve the video ID using get()
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                # Skip items without a valid video ID
                continue

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            title = item["snippet"].get("title", "No title")
            channel = item["snippet"].get("channelTitle", "Unknown channel")
            description = item["snippet"].get("description") or "No description available."

            result.append(
                f"*{title}*\n_Channel_: {channel}\n{description}\n<{video_url}|Watch here>\n"
            )

        return "\n".join(result) if result else "No videos found."
    except Exception as e:
        print(f"An error occurred while displaying video details: {e}")
        return "An error occurred while fetching video details."


def background_process(query, response_url):
    """
    Handles the query processing in the background to avoid timeouts.
    """
    try:
        # Fetch video results
        video_items = search_youtube(query)

        if not video_items:
            final_response = "Sorry, no results found on YouTube."
        else:
            # Get a formatted string of video details
            final_response = display_video_details(video_items)

        # Send the result back to Slack
        payload = {"text": final_response}
        requests.post(response_url, json=payload)
    except Exception as e:
        payload = {"text": f"An error occurred: {str(e)}"}
        requests.post(response_url, json=payload)


@flask_app.route("/youtubebot", methods=["POST"])
def handle_websearch():
    """
    Handles the /youtubebot Slack command.
    """
    data = request.form
    query = data.get("text", "").strip()
    response_url = data.get("response_url")

    # Immediately acknowledge the command
    ack_response = {"text": "*Fetching your data...*", "response_type": "ephemeral"}
    requests.post(response_url, json=ack_response)

    # Run background process with query and response_url
    threading.Thread(target=background_process, args=(query, response_url)).start()

    return jsonify({"response_type": "ephemeral", "text": "Processing your request..."}), 200


def background_process_for_mentions(query: str, say: Any):
    """
    Handles the background process for @bot mentions.

    Args:
        query (str): The search query extracted from the Slack mention.
        say (callable): A function for sending a response back to Slack.
    """
    try:
        # Fetch YouTube results
        video_items = search_youtube(query)

        if not video_items:
            final_response = "Sorry, no results found on YouTube."
        else:
            final_response = display_video_details(video_items)

        # Send the response back to Slack
        say(final_response)
    except Exception as e:
        say(f"An error occurred: {str(e)}")


@app.event("app_mention")
def handle_mentions(body: Dict[str, Any], say: Any) -> None:
    """
    Handles @bot mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    # Extract the message text from the Slack event
    text2 = body["event"]["text"]

    # Remove the bot mention from the message to get the query
    mention = f"<@{SLACK_BOT_USER_ID}>"
    query = text2.replace(mention, "").strip()

    # Respond immediately to acknowledge the mention
    say("*Woof* Fetching your data...")

    # Process the query in a separate thread to avoid timeout
    threading.Thread(target=background_process_for_mentions, args=(query, say)).start()



@flask_app.route("/slack/events", methods=["POST"])
def slack_events() -> Tuple[str, int, Dict[str, str]]:
    """
    Handles Slack events via Flask.
    """
    data = request.json
    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return data.get("challenge"), 200, {"Content-Type": "text/plain"}
    # Process other event types via SlackRequestHandler
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5000)
