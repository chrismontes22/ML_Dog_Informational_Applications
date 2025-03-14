**Welcome to My Dogbot for Slack**

Experience with Slack is required to run this bot. In this folder I have provided the vector database and Sentence Transformer model used. The script requires you to create a new bot in the Slack developer site, then obtain the Slackbot token, the Slack signing secret, and the Slackbot User ID. First obtain the bot token and signing secret from the bot's development pages, then run the retrieve_userid.py script to get the User ID. Finally you will require a Google API key to use one of the Gemini models free; Dogbot uses Gemini 1.5.

Next set up a .env file with all the keys, as demonstrated below. 

	SLACK_BOT_TOKEN = <SLACK_BOT_TOKEN>
	SLACK_SIGNING_SECRET = <SLACK_SIGNING_SECRET>
	SLACK_BOT_USER_ID = <SLACK_BOT_USER_ID>
	GOOGLE_API_KEY = <GOOGLE_API_KEY>

Finally expose it to the web and connect it to slack You may have to invite the bot as well.

Basic rundown

1. Set up bot and fill basic info like descriptions

2. Set up scopes by going to OAuth & Permissions:

	app_mentions:read
	chat:write
	channels:history

3. Install to workspace under Install App

4. Go to Oauth and copy/save Oauth token

5. Set the following in a .evn file (no quotations):

	SLACK_BOT_TOKEN = "Replace with token. Can Be Found  in OAuth. Starts with xox"
	SLACK_SIGNING_SECRET = "Replace with token. Can be found under Basic info"
	SLACK_BOT_USER_ID = "Python will get this, look at retrieve_userid.py for more details"
	GOOGLE_API_KEY = "Need to go to Google's developer page"

6. Pull the repo, and with SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET set, run the retrieve_userid.py file to get SLACK_BOT_USER_ID.

7. With all the environmental variables, run dogbot.py either with Python or Gunicorn.

8. Expose to the web the desired port. Then under the Slack web menu, go to event subscriptions, enable events on, add the correct url* you are working with, and subscribe to app_mention event name.
*The correct URL will have '/slack/events' the end of the url you are using for the app.

10. Finally go to basic info -> install app -> reinstall to workspace