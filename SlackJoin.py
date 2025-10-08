# A utility script to make the Slack bot join a specific public channel.
# Before running, ensure the bot has the 'channels:join' and 'channels:read' permission scopes.

import os
import time
from dotenv import load_dotenv, find_dotenv

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    print("Error: slack_sdk not found. Please run: pip install slack_sdk")
    exit()

# Load environment variables from .env file
load_dotenv(find_dotenv())

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

def join_specific_channel(client: "WebClient", channel_name: str):
    """
    Finds a public channel by its name and joins it if the bot is not already a member.
    """
    if not client:
        print("Error: Slack client could not be initialized.")
        return

    print(f"Attempting to find and join channel: #{channel_name}...")
    
    try:
        # We need to find the channel ID from its name.
        cursor = None
        channel_id = None
        
        while True:
            # Fetch a page of public channels
            resp = client.conversations_list(
                limit=1000, 
                cursor=cursor, 
                types="public_channel",
                exclude_archived=True
            )

            for channel in resp.get("channels", []):
                if channel.get("name") == channel_name:
                    channel_id = channel["id"]
                    if channel.get("is_member"):
                        print(f"✅ Bot is already a member of #{channel_name}.")
                        return
                    break # Exit the for loop once channel is found
            
            if channel_id:
                break # Exit the while loop if we found the channel

            # Check if there are more pages
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        if not channel_id:
            print(f"❌ Error: Channel '{channel_name}' not found or is not a public channel.")
            return

        # Try to join the channel using the found ID
        client.conversations_join(channel=channel_id)
        print(f"✅ Successfully joined #{channel_name}")

    except SlackApiError as e:
        if e.response.status_code == 429: # Rate limited
            retry_after = int(e.response.headers.get("Retry-After", 2))
            print(f"Rate limited. Waiting for {retry_after} seconds and trying again...")
            time.sleep(retry_after)
            join_specific_channel(client, channel_name) # Retry the function
        else:
            print(f"⚠️ An API error occurred: {e.response['error']}")


if __name__ == "__main__":
    if not SLACK_BOT_TOKEN:
        print("Error: SLACK_BOT_TOKEN not found in your .env file.")
    else:
        # The specific channel you want the bot to join
        CHANNEL_TO_JOIN = "bot-test"
        
        # Initialize the Slack client
        slack_client = WebClient(token=SLACK_BOT_TOKEN)
        join_specific_channel(slack_client, CHANNEL_TO_JOIN)

