import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import find_dotenv, load_dotenv


# Load environment variables from .env file
load_dotenv(find_dotenv())
# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    slack_client = WebClient(token=SLACK_BOT_TOKEN)

    try:
        # Call the auth.test method
        response = slack_client.auth_test()
        bot_user_id = response["user_id"]
        print(f"Bot User ID: {bot_user_id}")
    except SlackApiError as e:
        print(f"Error: {e.response['error']}")

get_bot_user_id()