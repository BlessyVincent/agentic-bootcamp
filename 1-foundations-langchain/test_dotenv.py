from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Check if a variable is read
api_key = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY:", api_key)
