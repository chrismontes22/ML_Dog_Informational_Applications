import os
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import find_dotenv, load_dotenv
import time

# Load environment variables from .env file
load_dotenv(find_dotenv())
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
CHANNEL_ID = os.environ["CHANNEL_ID"]  

def get_bitcoin_price() -> float:
    """Use Coindesk API to get current BTC price."""
    url = 'https://data-api.coindesk.com/index/cc/v1/latest/tick?market=cadli&instruments=BTC-USD,ETH-USD&apply_mapping=true'
    response = requests.get(url)
    data = response.json()

    # Navigate to the nested structure to find BTC-USD VALUE
    price = data["Data"]["BTC-USD"]["VALUE"]
    return price

def post_to_slack(price: float) -> None:
    """Uses the Slack WebClient to send a message with the Bitcoin price to a specific channel."""
    client = WebClient(token=SLACK_BOT_TOKEN)
    try:
        response = client.chat_postMessage(
            channel=CHANNEL_ID,
            text=f"Current Bitcoin price in USD: ${price:.2f}"
        )
        print(f"Message posted to {response['channel']}")
    except SlackApiError as e:
        print(f"Error posting to Slack: {e.response['error']}")

if __name__ == "__main__":
    while True:
        bitcoin_price = get_bitcoin_price()
        post_to_slack(bitcoin_price)
        time.sleep(300)  
