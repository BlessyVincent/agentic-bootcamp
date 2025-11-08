"""Convert a single-paragraph restaurant review into JSON following a strict schema.

Rules enforced by this script:
- Output printed to STDOUT will be ONLY valid JSON (no Markdown, no extra text).
- If facts are missing in the review, fields will be empty string or null as appropriate.
- Rating, when present, must be a number between 0.0 and 5.0.
- price_range, when present, must be one of: low | mid | high. If unknown, it will be null.

Usage: python json-restaurant-review.py --review "The food was great..." --max_retries 2
If --review is omitted the script will read from stdin.
"""

from dotenv import load_dotenv
import argparse
import json
import sys
from typing import Optional, Literal

from pydantic import BaseModel, Field, ValidationError, validator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


class RestaurantReview(BaseModel):
    name: str = Field("", description="Restaurant name; empty string if unknown")
    cuisine: str = Field("", description="Cuisine, e.g., Indian, Italian; empty string if unknown")
    city: str = Field("", description="City where the restaurant is located; empty string if not mentioned")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Rating between 0.0 and 5.0; null if unknown")
    price_range: Optional[Literal["low", "mid", "high"]] = Field(
        None, description='One of "low", "mid", "high"; null if unknown'
    )

    @validator("price_range")
    def _check_price_range(cls, v):
        if v is None:
            return v
        if v not in ("low", "mid", "high"):
            raise ValueError("price_range must be one of: low, mid, high")
        return v


def build_prompt(format_instructions: str) -> ChatPromptTemplate:
    system = (
        "You are an assistant that extracts structured restaurant information from a single-paragraph review. "
        "Never fabricate facts. If a field is not mentioned, return an empty string for text fields or null for numeric/enum fields."
    )

    user = (
        "Extract the restaurant information from the REVIEW_TEXT below and return ONLY valid JSON that matches the provided format instructions. "
        "Do not include any explanation, markdown, or extra keys.\n\n"
        "REVIEW_TEXT: {review}\n\n"
        "Follow these strict rules:\n"
        "1) Output must be only valid JSON, no surrounding text.\n"
        "2) Do not invent or hallucinate details. If not present, use empty string or null as described.\n"
        "3) rating must be a number between 0.0 and 5.0 (use null if unknown).\n"
        "4) price_range, if present, must be one of: low | mid | high (use null if unknown).\n\n"
        "FORMAT INSTRUCTIONS:\n{format_instructions}"
    )

    return ChatPromptTemplate.from_messages([("system", system), ("user", user)])


def strip_markdown_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove triple-fence and optional language tag
        lines = s.splitlines()
        # drop first and last line
        return "\n".join(lines[1:-1]).strip()
    return s


def main():
    parser = argparse.ArgumentParser(description="Convert a restaurant review paragraph to JSON following a schema.")
    parser.add_argument("--review", type=str, help="Single-paragraph restaurant review. If omitted read from stdin.")
    parser.add_argument("--max_retries", type=int, default=2, help="Number of retries on invalid JSON/validation")
    args = parser.parse_args()

    review_text = args.review
    if not review_text:
        review_text = sys.stdin.read().strip()
        if not review_text:
            print("{}")
            return

    # Initialize LLM (expects credentials in env)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_output_tokens=1024)

    parser_out = PydanticOutputParser(pydantic_object=RestaurantReview)
    format_instructions = parser_out.get_format_instructions()

    prompt_template = build_prompt(format_instructions)

    last_error_hint = ""

    # We'll try initial + retries
    for attempt in range(args.max_retries + 1):
        if attempt == 0:
            prompt_text = prompt_template.format(review=review_text, format_instructions=format_instructions)
        else:
            # append a short hint asking to correct
            retry_user = (
                prompt_template.messages[1][1]
                + "\n\nThe previous output failed validation with error: "
                + last_error_hint
                + "\nPlease provide only the corrected JSON, nothing else."
            )
            retry_prompt = ChatPromptTemplate.from_messages([prompt_template.messages[0], ("user", retry_user)])
            prompt_text = retry_prompt.format(review=review_text, format_instructions=format_instructions)

        # invoke LLM
        try:
            response = llm.invoke(prompt_text)
            llm_output = response.content
        except Exception as e:
            # LLM invocation error -> write to stderr and treat as fatal
            print(json.dumps({}), end="")
            print(f"LLM invocation failed: {e}", file=sys.stderr)
            return

        llm_output = strip_markdown_fences(llm_output)

        # Quick JSON parse check
        try:
            parsed = json.loads(llm_output)
        except Exception as e_json:
            last_error_hint = f"JSON parse error: {e_json}"
            print(last_error_hint, file=sys.stderr)
            if attempt == args.max_retries:
                # final failure: output empty JSON object to stdout (must be valid JSON)
                print(json.dumps({}), end="")
                return
            continue

        # Use the pydantic parser to validate and coerce
        try:
            structured: RestaurantReview = parser_out.parse(llm_output)
        except ValidationError as e_val:
            last_error_hint = str(e_val)
            print(f"Validation error: {last_error_hint}", file=sys.stderr)
            if attempt == args.max_retries:
                print(json.dumps({}), end="")
                return
            continue

        # Success: print only the JSON to stdout
        print(structured.model_dump_json(by_alias=False, exclude_none=True))
        return


if __name__ == "__main__":
    main()
