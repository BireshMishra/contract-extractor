import json
from dotenv import load_dotenv
import anthropic
from pydantic import ValidationError

from .models import ContractSummary

load_dotenv()

client = anthropic.Anthropic()

with open("./contracts/sample1_saas_subscription.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

schema = json.dumps(ContractSummary.model_json_schema(), indent=2)

system_prompt = f"""You extract structured data from contracts.
Respond with JSON only: no prose, no markdown fences.
The JSON must match this schema exactly:

{schema}

Use null for any field whose value is not stated in the contract."""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=8000,
    system=system_prompt,
    messages=[
        {
            "role": "user",
            "content": f"Extract the contract summary from this contract:\n\n{text_content}\n"
        }
    ]
)

for block in response.content:
    if block.type == "text":
        print(block.text)
        try:
            validated_model = ContractSummary.model_validate_json(block.text)
            print("Validated Model:", validated_model)
        except ValidationError as e:
            print("Validation failed!")
            print("Raw reply:", block.text)
            print("Error details:", e)



