# prompt | llm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser
import json 

load_dotenv()

class ProductInfo(BaseModel):
    name: str = Field(..., description="Name of the product")
    category: str = Field(..., description="Category of the product, e.g., 'laptop', 'smartphone', 'headphones'")
    price_estimate: Optional[float] = Field(None, description="Estimated price of the product as a number. Use null if unknown.")

    pros : List[str] = Field(default_factory=list, description="List of pros of the product as bullet points")
    cons : List[str] = Field(default_factory=list, description="List of cons of the product as bullet points")

args = argparse.ArgumentParser(description="JSON Formatted Output with Pydantic")
args.add_argument("--product_name", type=str, required=True, help="Name of the product")
args.add_argument("--description", type=str, required=True, help="Description of the product")
args.add_argument("--max_retries", type=int, default=2, help="Maximum number of retries for validation")
parsed_args = args.parse_args()


llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.5,
                                max_output_tokens=2048)

parser = PydanticOutputParser(pydantic_object=ProductInfo)
format_instructions = parser.get_format_instructions()

# print("Format Instructions:")
# print(format_instructions)


prompt = ChatPromptTemplate.from_messages(
[
    ("system",
        """You are an information extraction assistant. Extract Clean product information from user input.
        if the field is unknown for the provided text, set it to null.
        Or an empty list (for collections), and never fabricate the facts.
    """),
        ("user",
        """
        Extract product information from the text below and format it as per the instructions:
        INPUT_NAME: {name}\n
        INPUT_DESCRIPTION: {description}\n
        Follow the format instructions carefully:\n
        1. Return ONLY VALID JSON that adheres to the format instructions.
        2. Do not include any commentary or markdown fences
        3. Adhere to JSON schema and field descriptions.
        {format_instructions}
        """
        )
]
)

chain = prompt | llm_gemini

variables = {
    "name": parsed_args.product_name,
    "description": parsed_args.description,
    "format_instructions": format_instructions
}

last_error_hint = "" # blank for first try

for attempt in range(parsed_args.max_retries + 1):
    if attempt ==0:
        # first attempt : vanilla invoke
        prompt_prepared = prompt.format(**variables)
        print("\nPrepared Prompt for Gemini: in attempt 1")
        print(prompt_prepared)
    else:
        # retry with error hint appended to user message
        retry_prompt = ChatPromptTemplate.from_messages(
            [ *prompt.messages,
             ("user",
              f"""The previous output was invalid JSON.
              Here is the error encountered: {last_error_hint}\n\n
              Please correct the output to adhere to the format instructions.
              Please return ONLY the corrected JSON that follows the exact schema.
              Do not include any extra text"""
             )])
        prompt_prepared = retry_prompt.format(**variables)

    # invoke the model
    response = llm_gemini.invoke(prompt_prepared)
    print(f"\nResponse from Gemini via LCEL (Attempt {attempt + 1}):")
    print(response.content)
    content = response.content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.split("\n")[1:-1])  # remove markdown fences
    
    try:
        _ = json.loads(content)  # quick check for valid JSON
    
    except Exception as e_json:
        last_error_hint = f"JSON Parsing Error: {str(e_json)}"
        print(f"Validation failed: {last_error_hint}")
        if attempt == parsed_args.max_retries:
            print("Max retries reached. Exiting.")
            break
        else:
            continue  # retry
    
    product: ProductInfo = parser.parse(content)

    print("\nParsed Product Information:")

    print(product.model_dump_json(indent=2))
    print("It was succcessful!")

    break  # exit the retry loop on success
