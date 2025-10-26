# prompt | llm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse
  # take environment variables from .env

load_dotenv()
args = argparse.ArgumentParser(description="Dynamic Prompt Template with Argparse")
args.add_argument("--topic", type=str, required=True, help="Topic of the explainer")
args.add_argument("--audience", type=str, required=True, help="Target audience")
args.add_argument("--style", type=str, required=True, help="Style of the explainer")
args.add_argument("--length", type=str, required=True, help="Length of the explainer")
parsed_args = args.parse_args()

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.5,
                                max_output_tokens=2048)

prompt = ChatPromptTemplate.from_messages(
[
    ("system",
        "You are in expert assistant. Adapt Explanations to the audience and style."
        "Prefer Short sentences and concrete examples when helpful."),
        ("user",
        """Write a {style} explainer on the topic of '{topic}' for the {audience} audience.
        Keep it {length}.
        Format: \n
        Opener: 1-2 lines \n
        Core: 2-3 Bullet points \n
        Bottom Line: Single sentece starting with 'Bottom line:'
        """
        )
]
)

chain = prompt | llm_gemini

variables = {
    "topic": parsed_args.topic,
    "audience": parsed_args.audience,
    "style": parsed_args.style,
    "length": parsed_args.length}

response = chain.invoke(variables)
print("Response from Gemini via LCEL:")
print(response.content)




# python 7-dynamic-prompt-template-argparse.py --topic "Agentic AI" --audience "product manager" --style "executive brief" --length "under 120 words"
# python 7-dynamic-prompt-template-argparse.py --topic "Agentic AI" --audience "generaitive ai engineers" --style "dev intense" --length "under 120 words"