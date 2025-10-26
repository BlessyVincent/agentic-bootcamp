
# prompt | llm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import argparse
import sys

# Load environment variables (e.g. GOOGLE_API_KEY or credentials required by the Google GenAI client)
load_dotenv()

parser = argparse.ArgumentParser(description="Dynamic Prompt Template with Argparse")
parser.add_argument("--city", type=str, required=True, help="Destination city")
parser.add_argument("--days", type=int, required=True, help="Number of days to stay (integer)")
parser.add_argument("--budget", type=str, required=True, help="Budget for the trip")
parser.add_argument("--traveler_type", type=str, required=True, help="Type of traveler")
args = parser.parse_args()

if args.days < 1:
    print("Error: --days must be >= 1", file=sys.stderr)
    sys.exit(2)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_output_tokens=2000,
)

# Use template placeholders so the prompt engine can substitute values when the chain runs.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Travel Recommender Agent.
Provide personalized travel itineraries based on user preferences.
Prefer concise and clear recommendations.
""",
        ),
        (
            "user",
            """Create a {days}-day travel itinerary for a trip to {city} with a budget of {budget}.
The traveler is a {traveler_type}.

Opener: 1–2 line intro about the city.

Itinerary → Day-by-day plan matching the number of days.

Tips → 2–3 short suggestions tailored to budget and traveler type.

Use concrete travel suggestions, not vague descriptions.
Itinerary must match exactly the number of days specified.
Have fun and be creative with your recommendations!
""",
        ),
    ]
)

chain = prompt | llm_gemini

variables = {
    "city": args.city,
    "days": args.days,
    "budget": args.budget,
    "traveler_type": args.traveler_type,
}

try:
    response = chain.invoke(variables)
    # Some LCEL/LLM wrappers put the text on `response.content`, others return a string directly.
    content = getattr(response, "content", response)
    print("Response from Gemini via LCEL:")
    print(content)
except Exception as e:
    print("Error invoking chain:", e, file=sys.stderr)
    sys.exit(1)