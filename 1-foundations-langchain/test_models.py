import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Configure the library
genai.configure(api_key=api_key)

# List available models
for model in genai.list_models():
    print(f"Name: {model.name}")
    print(f"Description: {model.description}")
    print(f"Generation Methods: {model.supported_generation_methods}")
    print("-" * 50)